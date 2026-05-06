import os
from collections import deque
from pathlib import Path
from typing import List, Dict

import imageio
import matplotlib
import matplotlib.colors as mcolors
import numpy as np
import pandas as pd
import pytorch_lightning as L
import rustworkx as rx
import torch
from hydra.utils import instantiate
from loguru import logger
from matplotlib import cm
from matplotlib import pyplot
from matplotlib.backends.backend_agg import FigureCanvasAgg
from torch.multiprocessing import Queue
from torch.nn.functional import binary_cross_entropy_with_logits
from torchrl.data import ReplayBuffer, ListStorage, PrioritizedSampler
from tqdm import trange

from src.definitions import EnvironmentConfig

torch.multiprocessing.set_start_method("fork", force=True)

from src.data import GraphDataset
from src.logging import configure_logging
from src.util import batch_fast, EWMA


class Trajectory:
    def __init__(self, dataset: GraphDataset, train: bool, target=None):
        self.dataset = dataset
        self.train = train

        self.predicted_graph_pt = None
        self.predicted_edge_mask = None

        self.successors_rx = None
        self.predictions = []
        self.successor_q = None
        self.history_q = deque() if not self.train else None

        self.done = False
        self.correct = False

        self.step_number = 0
        self.step_recording = dict()
        self.step_successors = torch.zeros(100) - 1

        self.target_graph_rx = self.dataset.get_item() if target is None else target
        self.target_graph_pt = self.dataset.rx_to_pt(self.target_graph_rx)

        # using half results in a good mix of early and later states in the
        # beginning of training
        if self.train and np.random.rand() < 0.5:
            self.predicted_graph_rx = self.get_target_subgraph()
        else:
            self.predicted_graph_rx = rx.PyGraph()

        self.action_filter = set()

        self.step()

    def get_target_subgraph(self, min_nodes=1):
        size = np.random.randint(min_nodes, self.target_graph_rx.num_nodes())
        return self.dataset.fireforest_sample(self.target_graph_rx, size, (0.2, 0.9))

    def action(self):
        graph_predictions, mask_predictions = zip(*self.predictions)

        graph_predictions = torch.stack(graph_predictions)

        idx = graph_predictions.argmax()

        self.predicted_graph_rx = self.successors_rx[idx]
        self.predicted_graph_pt = self.dataset.get_single_sample(
            self.target_graph_rx, self.predicted_graph_rx
        )
        self.predicted_edge_mask = mask_predictions[idx]

        assert len(self.successors_rx) == len(graph_predictions)

        if self.train or self.step_number < 10:
            return

        for successor, prob in zip(self.successors_rx, graph_predictions.sigmoid()):
            action = successor.attrs["action"]
            keep = prob > 0.1

            if keep or action is None:
                continue

            self.action_filter.add(action)

    def step(self):
        self.step_number += 1

        if self.train or self.predicted_edge_mask is None:
            edges_mask = self.dataset.get_edge_mask(
                self.target_graph_rx, self.predicted_graph_rx
            )
        else:
            edges_mask = self.predicted_edge_mask.sigmoid()

        edges_mask = self.dataset.get_edge_filter_from_mask(edges_mask)

        nodes_mask = self.action_filter

        self.successors_rx = self.dataset.successors_rx(
            self.predicted_graph_rx,
            edges_mask=edges_mask,
            nodes_mask=nodes_mask,
            terminal=True,
        )

        self.step_successors[self.step_number] = len(self.successors_rx)

        self.successor_q = deque(self.successors_rx)
        self.predictions.clear()

    def next(self):
        rx_graph = self.successor_q.popleft()
        pt_graph = self.dataset.get_single_sample(self.target_graph_rx, rx_graph)
        if not self.train:
            self.history_q.append(pt_graph)
        return pt_graph

    def check_done(self):
        valid = (self.predicted_graph_pt["label"] > 0).item()
        terminal = (self.predicted_graph_pt["terminal"] > 0).item()
        same = self.dataset.is_same_size(self.predicted_graph_rx, self.target_graph_rx)

        self.correct = valid and same
        self.done = not valid or terminal

    def update(self, graph_prediction, mask_prediction=None):
        self.predictions.append((graph_prediction, mask_prediction))

        if len(self.predictions) == len(self.successors_rx):
            assert len(self.successor_q) == 0, "queue not empty"
            self.action()
            self.step()
            self.check_done()

    def record(self, graph_pred, mask_pred):
        if self.step_number not in self.step_recording:
            self.step_recording[self.step_number] = {
                "successors_rx": self.successors_rx,
                "y_true_graph": [],
                "y_true_mask": [],
                "terminals": [],
                "y_pred_graph": [],
                "y_pred_mask": [],
            }

        g = self.history_q.popleft()

        self.step_recording[self.step_number]["y_true_graph"].append(g["label"])
        self.step_recording[self.step_number]["y_true_mask"].append(g["edge_mask"])
        self.step_recording[self.step_number]["terminals"].append(g["terminal"])
        self.step_recording[self.step_number]["y_pred_graph"].append(graph_pred.item())
        self.step_recording[self.step_number]["y_pred_mask"].append(mask_pred.tolist())

    def top_k(self, y_pred, y_true):
        order = y_pred.argsort(descending=True)
        labels = y_true[order]
        valid = labels.sum()
        frac = valid / (labels.nonzero()[-1] + 1) if valid > 0 else 0
        return frac

    def summary(self):
        all_successors = []
        all_losses = []
        all_topk = []

        for step in self.step_recording.values():
            all_successors.append(len(step["successors_rx"]))

            y_true_graph = torch.tensor(step["y_true_graph"])
            y_pred_graph = torch.tensor(step["y_pred_graph"])

            assert y_pred_graph.shape == y_pred_graph.shape

            loss = binary_cross_entropy_with_logits(y_pred_graph, y_true_graph)
            all_losses.append(loss)
            all_topk.append(self.top_k(y_pred_graph, y_true_graph))

        successors = torch.tensor(all_successors).float()
        losses = torch.tensor(all_losses).float()
        top_k = torch.tensor(all_topk).float()

        return {
            "correct": torch.tensor(self.correct).float().item(),
            "length_true": torch.tensor(self.target_graph_rx.num_edges()).item() + 1,
            "length_pred": torch.tensor(len(self.step_recording)).item() - 1,
            "successors_sum": successors.sum().item(),
            "successors_avg": successors.mean().item(),
            "loss": losses.mean().item(),
            "top_k": top_k.mean().item(),
            "step_successors": self.step_successors.tolist(),
        }


class TrajectoryPlotter:
    def __init__(
        self,
        trajectory: Trajectory,
        dataset: GraphDataset,
        directory: str | Path,
    ):
        self.trajectory = trajectory
        self.dataset = dataset
        self.directory = directory
        self.max_entries = 70
        self.max_per_table = 35
        self.max_tables = self.max_entries // self.max_per_table

    def round(self, values, d=3):
        return list(map(lambda x: round(x, d), values))

    def norm(self, array, cvt):
        if cvt:
            array = array.tolist()

        array = self.round(array)
        return mcolors.Normalize(vmin=min(array), vmax=max(array)), array

    def plot_label_masks(self, ax, masks_pred, masks_true):
        dist_pred_raw = masks_pred
        dist_pred_prob = masks_pred.sigmoid()
        dist_true = masks_true
        idx = (dist_pred_prob + dist_true).argsort(descending=True)

        dist_pred_raw = self.round(dist_pred_raw[idx].tolist())[: self.max_per_table]
        dist_pred_prob = self.round(dist_pred_prob[idx].tolist())[: self.max_per_table]
        dist_true = self.round(dist_true[idx].tolist())[: self.max_per_table]

        col_data = {
            3: self.norm(dist_pred_raw, cvt=False),
            4: self.norm(dist_pred_prob, cvt=False),
            5: self.norm(dist_true, cvt=False),
        }

        connections = [
            self.dataset.generator.get_connection(*m) for m in self.dataset.mask
        ]
        connections = [connections[i] for i in idx]

        connection_type = connections[0]["type"]

        data = [
            m["connection"] + (r, p, t)
            for m, r, p, t in zip(connections, dist_pred_raw, dist_pred_prob, dist_true)
        ]

        cmap = cm.Purples  # noqa

        columns = [
            "u",
            "-",
            "v",
            "$y$",
            r"$\sigma$",
            r"$\checkmark$",
        ]

        table = ax.table(
            colLabels=columns,
            cellText=data,
            colWidths=[0.05] * 3 + [0.1] * 3,
            loc="center",
            cellLoc="center",
        )

        table.auto_set_font_size(False)
        table.set_fontsize(10)

        ax.axis("off")

        for (row, col), cell in table.get_celld().items():
            if connection_type == "color" and 0 <= col <= 2:
                color = connections[row - 1]["connection"][col]
                cell.set_facecolor(color)
                cell.get_text().set_visible(False)

            if col not in col_data:
                continue

            col_norm, col_values = col_data[col]

            row_value = col_values[row - 1]
            row_norm_v = col_norm(row_value)

            cell.set_facecolor(cmap(row_norm_v))
            if row_norm_v > 0.5:
                cell.get_text().set_color("white")

    def plot_table_labels(
        self,
        axes: matplotlib.axes.Axes,  # noqa
        preds: torch.Tensor,
        labels: torch.Tensor,
        correct: bool,
        idx: int,
        connections: List[Dict],
    ):
        all_preds = preds.cpu()
        all_probs = all_preds.sigmoid().cpu()
        all_labels = labels.cpu()
        n_correct = int(all_labels.sum().item())

        conn_types, connections = zip(
            *[(c["type"], c["connection"]) for c in connections]
        )
        us, es, vs = zip(*connections)
        color = "#4287f5"

        n = len(all_preds)
        all_ids = list(range(n))
        all_indices = list(reversed(np.argsort(all_preds)))[: self.max_entries]

        cmap = cm.Purples  # noqa

        if correct:
            cmap = cm.Greens  # noqa

        col_data = {
            0: self.norm(all_ids, cvt=False),
            1: self.norm(all_preds, cvt=True),
            2: self.norm(all_probs, cvt=True),
            3: self.norm(all_labels, cvt=True),
            4: (None, us),
            5: (None, es),
            6: (None, vs),
        }

        all_columns_data = [v[1] for k, v in col_data.items()]
        columns = [
            f"[{n}]",
            r"$y$",
            r"$\sigma$",
            r"$\checkmark$" + f" ({n_correct})",
            "u",
            "-",
            "v",
        ]

        size = self.max_per_table

        for i, ax in enumerate(axes):
            table_indices = all_indices[i * size : i * size + size]

            if not table_indices:
                return

            (ids, labels, preds, fracs, us, es, vs) = [
                [x[i] for i in table_indices] for x in all_columns_data
            ]

            empty = [""] * len(us)

            # connections don't have a cell text, leave them empty
            cell_text = list(zip(ids, labels, preds, fracs, empty, empty, empty))

            frac = 0.8

            table = ax.table(
                colLabels=columns,
                cellText=cell_text,
                cellLoc="center",
                colWidths=[frac / 4] * 4 + [(1.0 - frac) / 3] * 3,
                loc="center",
            )

            table.auto_set_font_size(False)
            table.set_fontsize(10)

            if idx in ids:
                i = ids.index(idx) + 1
                table[(i, 0)].set_facecolor(color)

            for (row, col), cell in table.get_celld().items():
                # must be placed at the beginning for proper header handling
                if 4 <= col <= 6:

                    if row == 0:
                        continue

                    cs = {4: us, 5: es, 6: vs}
                    bg = cs[col][row - 1]
                    if conn_types[0] == "color":
                        cell.set_facecolor(bg)

                    if conn_types[0] == "text":
                        cell.get_text().set_text(bg)

                    continue

                if row == 0 or col == 0:
                    continue

                col_norm, col_values = col_data[col]

                col_values = [col_values[i] for i in table_indices]

                row_value = col_values[row - 1]
                row_norm_v = col_norm(row_value)

                cell.set_facecolor(cmap(row_norm_v))

                if row_norm_v > 0.5:
                    cell.get_text().set_color("white")

    def plot_step(self, t: int, image, step, correct: bool):
        successors = step["successors_rx"]
        y_pred_graph = torch.tensor(step["y_pred_graph"])
        y_true_graph = torch.cat(step["y_true_graph"])

        connections = [g.attrs["connection"] for g in successors]
        idx = y_pred_graph.argmax().item()
        graph = successors[idx]
        y_pred_mask = torch.tensor(step["y_pred_mask"])[idx]
        y_true_mask = torch.stack(step["y_true_mask"])[idx]

        n_cols = 3 + self.max_tables

        fig, axes = pyplot.subplots(2, n_cols, figsize=(28, 16))

        for ax in axes.ravel():
            ax.axis("off")

        axes[0, 0].imshow(image)

        self.dataset.generator.draw_graph(self.dataset.rx_to_nx(graph), axes[0, 1])
        axes[0, 1].set_title(f"step: {t}, correct: {correct}")

        self.plot_table_labels(
            axes[0, 2:-1],
            y_pred_graph,
            y_true_graph,
            correct,
            idx,
            connections,
        )

        self.plot_label_masks(axes[0, -1], y_pred_mask, y_true_mask)

        loss = binary_cross_entropy_with_logits(
            y_pred_graph, y_true_graph, reduction="none"
        )
        loss_idx = torch.argsort(loss, descending=True)
        n_plots = min(len(loss_idx), n_cols)

        for ax, idx in zip(axes[1, :n_plots], loss_idx):
            frame = self.dataset.generator.get_frame(
                self.dataset.rx_to_nx(successors[idx])
            )
            header = f"idx: {idx}, loss: {loss[idx]:.4f}"
            ax.imshow(frame)
            ax.set_title(header, pad=50)

        fig.tight_layout()

        canvas = FigureCanvasAgg(fig)
        s, (width, height) = canvas.print_to_buffer()
        frame = np.frombuffer(s, np.uint8).reshape((height, width, 4))
        return frame

    def plot(self, filename: Path):
        file = os.path.join(self.directory, filename)

        target_graph_nx = self.dataset.rx_to_nx(self.trajectory.target_graph_rx)
        image = self.dataset.generator.get_frame(target_graph_nx)

        steps = list(self.trajectory.step_recording.values())
        frames = [
            self.plot_step(i, image, step, False) for i, step in enumerate(steps[:-1])
        ]
        frames += [
            self.plot_step(len(steps) - 1, image, steps[-1], self.trajectory.correct)
        ]

        with imageio.get_writer(file, fps=10) as writer:
            for frame in frames:
                writer.append_data(frame)

        pyplot.close("all")


def rollout_train(
    worker_id: int,
    worker_total: int,
    in_q: Queue,
    out_q: Queue,
    environment: EnvironmentConfig,
    total_batch_size: int,
    buffer_batch_size: int,
    buffer_min_size: int,
    buffer_max_size: int,
    buffer_multiplier: int,
):
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)

    allowed = sorted(os.sched_getaffinity(0))
    core = allowed[worker_id % len(allowed)]
    os.sched_setaffinity(0, {core})
    logger.info(f"worker {worker_id} pinning to core {core}")

    configure_logging(filename=None, rank=worker_id)

    generator = instantiate(environment, train=True)
    dataset = GraphDataset(generator)
    dataset.get_item()

    buffer = ReplayBuffer(
        storage=ListStorage(max_size=buffer_max_size),
        sampler=PrioritizedSampler(alpha=1.0, beta=1.0, max_capacity=buffer_max_size),
        batch_size=buffer_batch_size,
        collate_fn=lambda x: x,
    )

    buffer_samples = 0
    prio_ewma = EWMA(decay=0.999)

    n_parallel_rollouts = int(1.1 * total_batch_size)

    logger.info(f"worker {worker_id+1}/{worker_total} init")
    trajectories = [
        Trajectory(dataset, train=True)
        for _ in trange(n_parallel_rollouts, disable=worker_id != 0)
    ]

    logger.info(f"worker {worker_id+1}/{worker_total} init done")

    while True:
        buffer_active = buffer_samples > 1 and len(buffer) >= buffer_min_size

        n_new_samples = total_batch_size - buffer_active * buffer_batch_size

        current_idx = torch.randperm(n_parallel_rollouts)[:n_new_samples]
        current_trajectories = [trajectories[i] for i in current_idx]

        current_targets = [t.target_graph_pt for t in current_trajectories]
        current_subgraphs = [t.next() for t in current_trajectories]

        buffer_idx = None

        if buffer_active:
            old_batch, old_info = buffer.sample(return_info=True)
            buffer_idx = old_info["index"]

            old_targets, old_samples = zip(*old_batch)

            current_targets += list(old_targets)
            current_subgraphs += list(old_samples)

            buffer_samples -= len(old_targets)

        subgraphs = [x["graph"] for x in current_subgraphs]

        subgraphs = batch_fast(subgraphs)

        terminals = torch.cat([x["terminal"] for x in current_subgraphs])
        y_true_graph = torch.cat([x["label"] for x in current_subgraphs])
        y_true_edge_mask = torch.stack([x["edge_mask"] for x in current_subgraphs])
        times = torch.cat([x["time"] for x in current_subgraphs])

        targets = None
        if generator.task == "image":
            targets = torch.stack([x["image"] for x in current_subgraphs])
        if generator.task == "fingerprint":
            targets = torch.stack([x["fingerprint"] for x in current_subgraphs])
        if generator.task == "graph":
            targets = batch_fast(current_targets)

        batch = {
            "targets": targets,
            "subgraphs": subgraphs,
            "terminals": terminals.unsqueeze(-1),
            "y_graph": y_true_graph.unsqueeze(-1),
            "y_edge_mask": y_true_edge_mask,
            "times": times,
        }

        out_q.put(batch)

        predictions, prios = in_q.get()

        prio_ewma.update(prios.mean())

        if buffer_active:
            new_predictions = predictions[:-buffer_batch_size]
            new_prios = prios[:-buffer_batch_size]
            old_prios = prios[-buffer_batch_size:]

            buffer.update_priority(buffer_idx, old_prios)
        else:
            new_predictions = predictions
            new_prios = prios

        for idx, target, subgraph, pred, prio in zip(
            current_idx, current_targets, current_subgraphs, new_predictions, new_prios
        ):
            trajectories[idx].update(pred)

            if prio > prio_ewma.compute():
                buffer_idx_new = buffer.add((target, subgraph))
                buffer.update_priority(buffer_idx_new, prio)
                buffer_samples += buffer_multiplier

        trajectories = [t for t in trajectories if not t.done]

        diff = n_parallel_rollouts - len(trajectories)

        for _ in range(diff):
            trajectories.append(Trajectory(dataset, train=True))


def rollout_test(
    request_q: Queue,
    response_q: Queue,
    environment: EnvironmentConfig,
    directory: Path,
    batch_size: int,
    return_results: bool = False,
):
    torch.set_num_threads(1)

    trajectories = []
    active_trajectories = set()

    epoch = 0

    generator = instantiate(environment, train=False)
    dataset = GraphDataset(generator)

    while True:
        request = request_q.get()

        if request["type"] == "reset":
            epoch += 1
            trajectories = [
                Trajectory(dataset, train=False, target=t) for t in request["targets"]
            ]
            active_trajectories = set(range(len(trajectories)))
            continue

        if request["type"] == "batch":
            current_batch_ids = []
            current_traj_ids = []
            current_targets = []
            current_images = []
            current_fingerprints = []
            current_samples = []

            for batch_id, traj_id in enumerate(active_trajectories):
                if len(current_samples) == batch_size:
                    break

                trajectory = trajectories[traj_id]
                current_targets.append(trajectory.target_graph_pt)

                if "image" in trajectory.target_graph_rx.attrs:
                    current_images.append(trajectory.target_graph_rx.attrs["image"])

                if "fingerprint" in trajectory.target_graph_rx.attrs:
                    current_fingerprints.append(
                        trajectory.target_graph_rx.attrs["fingerprint"]
                    )

                while len(trajectory.successor_q) > 0:
                    subgraph = trajectory.next()
                    current_batch_ids.append(batch_id)
                    current_traj_ids.append(traj_id)
                    current_samples.append(subgraph)

            if current_batch_ids:
                batch_ids = torch.tensor(current_batch_ids)
                traj_ids = torch.tensor(current_traj_ids)
                subgraphs = batch_fast([x["graph"] for x in current_samples])
                terminals = torch.cat([x["terminal"] for x in current_samples])
                times = torch.cat([x["time"] for x in current_samples])

                targets = None
                if generator.task == "image":
                    targets = torch.stack(current_images)
                if generator.task == "fingerprint":
                    targets = torch.stack(current_fingerprints)
                if generator.task == "graph":
                    targets = batch_fast(current_targets)

                batch = {
                    "type": "batch",
                    "batch_ids": batch_ids,
                    "traj_ids": traj_ids,
                    "targets": targets,
                    "subgraphs": subgraphs,
                    "terminals": terminals.unsqueeze(-1),
                    "times": times,
                }

                response_q.put(batch)
            else:
                response_q.put({"type": "done"})

            continue

        if request["type"] == "prediction":
            traj_ids = request["traj_ids"].tolist()
            y_pred_graph = request["y_pred_graph"]
            y_pred_mask = request["y_pred_mask"]

            assert len(traj_ids) == len(y_pred_graph) == len(y_pred_mask)
            for traj_id, graph_pred, mask_pred in zip(
                traj_ids, y_pred_graph, y_pred_mask
            ):
                t = trajectories[traj_id]
                # must be before update, otherwise record is done for next step
                t.record(graph_pred, mask_pred)
                t.update(graph_pred, mask_pred)

                if t.done:
                    active_trajectories.remove(traj_id)
            continue

        if request["type"] == "summary":
            summaries = [t.summary() for t in trajectories]
            df = pd.DataFrame(summaries)

            summary = {
                "type": "summary",
                "correct": df["correct"].mean(),
                "length_min_pred": df["length_pred"].min(),
                "length_mean_pred": df["length_pred"].mean(),
                "length_max_pred": df["length_pred"].max(),
                "length_min_true": df["length_true"].min(),
                "length_mean_true": df["length_true"].mean(),
                "length_max_true": df["length_true"].max(),
                "top_k": df["top_k"].mean(),
                "successors_sum": df["successors_sum"].sum(),
                "successors_avg": df["successors_avg"].mean(),
                "eval_loss": df["loss"].mean(),
            }

            if return_results:
                summary["step_successors"] = df["step_successors"]

            summary = {
                k: (
                    torch.tensor(v, dtype=torch.float32)
                    if not isinstance(v, (str, list))
                    else v
                )
                for k, v in summary.items()
            }

            response_q.put(summary)

            continue

        if request["type"] == "plot":
            correct = [t for t in trajectories if t.correct]
            incorrect = [t for t in trajectories if not t.correct]

            if correct:
                correct.sort(key=lambda t: t.step_number)
                TrajectoryPlotter(correct[0], dataset, directory).plot(
                    directory / f"epoch_{epoch}_correct_shortest.mp4"
                )
                TrajectoryPlotter(correct[-1], dataset, directory).plot(
                    directory / f"epoch_{epoch}_correct_longest.mp4"
                )

            if incorrect:
                incorrect.sort(key=lambda t: t.step_number)
                TrajectoryPlotter(incorrect[0], dataset, directory).plot(
                    directory / f"epoch_{epoch}_incorrect_shortest.mp4"
                )
                TrajectoryPlotter(incorrect[-1], dataset, directory).plot(
                    directory / f"epoch_{epoch}_incorrect_longest.mp4"
                )
            continue

        raise RuntimeError(f"unexpected request type {request["type"]}")
