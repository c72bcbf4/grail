import itertools
import os
import random
import sys
import traceback
from collections import defaultdict
from pathlib import Path

import psutil
import pytorch_lightning as L
import torch
from hydra.utils import instantiate
from lightning_fabric import Fabric
from loguru import logger
from matplotlib import pyplot
from pytorch_optimizer import Lamb
from torch import nn
from torchmetrics.classification import BinaryF1Score, BinaryStatScores
from torchvision.ops import sigmoid_focal_loss
from tqdm import trange

from src.data import GraphDataset
from src.definitions import AppConfig
from src.logging import configure_logging
from src.monitoring import Timer
from src.nn import LatentToGraphDecoder
from src.plot import plot_training
from src.rollout import rollout_train, rollout_test
from src.util import (
    CosineWarmupDecayLR,
    count_parameters,
    AutoResetMeanMetric,
    Batcher,
    SystemMonitor,
    configure_weight_decay,
    label_smoothing,
)


def adjust_config_to_world_size(conf: AppConfig, world_size: int):
    assert conf.training.buffer_batch_size <= conf.training.total_batch_size // 2

    conf.training.samples_per_epoch //= world_size
    conf.training.rollouts_per_epoch //= world_size
    conf.training.total_batch_size //= world_size

    conf.training.buffer_batch_size //= world_size
    conf.training.buffer_min_size //= world_size
    conf.training.buffer_max_size //= world_size

    conf.optimizer.lr_warmup_samples //= world_size
    conf.optimizer.lr_total_samples //= world_size
    return conf


def train(fabric: Fabric, conf: AppConfig):
    try:
        pid = os.getpid()
        proc = psutil.Process(pid)

        pyplot.rcParams.update({"figure.max_open_warning": 100})

        conf = adjust_config_to_world_size(conf, fabric.world_size)

        conf.environment.task = conf.encoder_target.task

        directory = Path(conf.output_dir)
        rank = fabric.global_rank if conf.batch_size == 1 else conf.batch_idx

        if fabric.is_global_zero:
            directory.mkdir(parents=True, exist_ok=True)

        configure_logging(
            filename=directory / "log.log",
            rank=rank,
        )

        device = str(fabric.device)

        generator_train = instantiate(conf.environment, train=True)

        dataset = GraphDataset(generator_train)

        directory_test = directory / "eval"
        directory_test.mkdir(parents=True, exist_ok=True)

        hidden_activation = nn.SiLU()

        model = LatentToGraphDecoder(
            dataset=dataset,
            encoder_target=conf.encoder_target,
            encoder_query=conf.encoder_query,
            decoder=conf.decoder,
            activation=hidden_activation,
        )

        model = nn.SyncBatchNorm.convert_sync_batchnorm(model)

        parameters = [
            f"params.encoder_target={count_parameters(model.encoder_target):,}",
            f"params.encoder_query={count_parameters(model.encoder_query):,}",
            f"params.decoder_graph={count_parameters(model.decoder_graph):,}",
            f"params.decoder_edge={count_parameters(model.decoder_edge):,}",
            f"params.total={count_parameters(model):,}",
            f"meta.task={conf.environment.task}",
        ]
        conf.overrides.extend(parameters)

        optimizer = Lamb(
            configure_weight_decay(model, conf.optimizer.weight_decay),
            lr=conf.optimizer.lr_start or conf.optimizer.lr_max,
            betas=(conf.optimizer.beta_1, conf.optimizer.beta_2),
            eps=conf.optimizer.eps,
        )

        model, optimizer = fabric.setup(model, optimizer)

        model.mark_forward_method("encode_target")
        model.mark_forward_method("encode_query")
        model.mark_forward_method("decode")

        scheduler = CosineWarmupDecayLR(
            optimizer,
            lr_start=conf.optimizer.lr_start or conf.optimizer.lr_max,
            lr_max=conf.optimizer.lr_max,
            lr_end=conf.optimizer.lr_end or conf.optimizer.lr_max,
            total_steps=conf.optimizer.lr_total_samples
            // conf.training.total_batch_size,
            warmup_steps=conf.optimizer.lr_warmup_samples
            // conf.training.total_batch_size,
        )

        metric_loss_graph_pred = AutoResetMeanMetric().to(device)
        metric_loss_edge_pred = AutoResetMeanMetric().to(device)

        metric_logits_graph_pos = AutoResetMeanMetric().to(device)
        metric_logits_graph_neg = AutoResetMeanMetric().to(device)
        metric_logits_edge_pos = AutoResetMeanMetric().to(device)
        metric_logits_edge_neg = AutoResetMeanMetric().to(device)

        metric_pos_graph = AutoResetMeanMetric().to(device)
        metric_pos_edges = AutoResetMeanMetric().to(device)

        metric_f1_graph = BinaryF1Score().to(device)
        metrics_binary = BinaryStatScores().to(device)
        metric_f1_edges = BinaryF1Score().to(device)

        metric_grad = AutoResetMeanMetric().to(device)

        training_logs = []

        timer = Timer(disable=not fabric.is_global_zero)

        logger.info(f"starting {conf.training.n_train_workers} training workers")
        logger.info(f"starting {conf.training.n_test_workers} testing workers")

        mk_q = lambda: torch.multiprocessing.Queue(maxsize=10)

        train_queues = [(mk_q(), mk_q()) for _ in range(conf.training.n_train_workers)]
        test_queues = [(mk_q(), mk_q()) for _ in range(conf.training.n_test_workers)]

        train_batcher = Batcher(train_queues, device)
        if conf.batch_size == 1:
            train_worker_offset = fabric.global_rank * conf.training.n_train_workers
            train_workers_total = fabric.world_size * conf.training.n_train_workers
        else:
            train_worker_offset = conf.batch_idx * conf.training.n_train_workers
            train_workers_total = conf.training.n_train_workers

        train_workers = [
            torch.multiprocessing.Process(
                target=rollout_train,
                args=(
                    train_worker_offset + i,
                    train_workers_total,
                    in_q,
                    out_q,
                    conf.environment,
                    conf.training.total_batch_size,
                    conf.training.buffer_batch_size,
                    conf.training.buffer_min_size // conf.training.n_train_workers,
                    conf.training.buffer_max_size // conf.training.n_train_workers,
                    conf.training.buffer_multiplier,
                ),
                daemon=True,
            )
            for i, (in_q, out_q) in enumerate(train_queues)
        ]

        test_workers = [
            torch.multiprocessing.Process(
                target=rollout_test,
                args=(
                    req_q,
                    resp_q,
                    conf.environment,
                    directory_test,
                    conf.training.total_batch_size,
                ),
                daemon=True,
            )
            for i, (req_q, resp_q) in enumerate(test_queues)
        ]

        for w in train_workers + test_workers:
            w.start()

        logger.info("starting workers done")

        sys_monitor = SystemMonitor().start()

        for epoch in range(
            conf.training.samples_per_run // conf.training.samples_per_epoch
        ):
            timer.start("epoch")
            timer.start("train")
            model.train()

            time_logs = []

            for _ in trange(
                conf.training.samples_per_epoch // conf.training.total_batch_size,
                disable=not fabric.is_global_zero,
                leave=False,
                file=sys.stdout,
            ):
                response, in_q = train_batcher.get()

                y_true_graph = response["y_graph"]
                y_true_edge = response["y_edge_mask"]

                targets = response["targets"]

                targets_emb = model.encode_target(targets)
                subgraphs_emb = model.encode_query(response["subgraphs"], targets_emb)

                y_pred_graph, y_pred_edge = model.decode(
                    targets_emb, subgraphs_emb, response["terminals"]
                )

                loss_graph_pred = sigmoid_focal_loss(
                    y_pred_graph,
                    label_smoothing(y_true_graph, conf.optimizer.label_smoothing),
                    alpha=-1,
                    gamma=conf.optimizer.gamma,
                    reduction="none",
                )
                loss_edge_pred = sigmoid_focal_loss(
                    y_pred_edge,
                    label_smoothing(y_true_edge, conf.optimizer.label_smoothing),
                    alpha=-1,
                    gamma=conf.optimizer.gamma,
                    reduction="none",
                )

                prio = (y_pred_graph.sigmoid() - y_true_graph).abs()

                in_q.put(
                    (
                        y_pred_graph.detach().reshape(-1).cpu(),
                        prio.detach().reshape(-1).cpu(),
                    )
                )

                loss = loss_graph_pred.mean() + loss_edge_pred.mean()

                optimizer.zero_grad()
                fabric.backward(loss)
                grad_norm = fabric.clip_gradients(model, optimizer, max_norm=1.0)
                optimizer.step()
                scheduler.step()

                with torch.no_grad():
                    metric_loss_graph_pred.update(loss_graph_pred)
                    metric_loss_edge_pred.update(loss_edge_pred)

                    metric_logits_graph_pos.update(y_pred_graph[y_true_graph == 1])
                    metric_logits_graph_neg.update(y_pred_graph[y_true_graph == 0])

                    metric_logits_edge_pos.update(y_pred_edge[y_true_edge == 1])
                    metric_logits_edge_neg.update(y_pred_edge[y_true_edge == 0])

                    time_logs.append(response["times"])

                    metric_pos_graph.update(y_true_graph)
                    metric_pos_edges.update(y_true_edge)

                    metric_f1_graph.update(y_pred_graph, y_true_graph)
                    metrics_binary.update(y_pred_graph, y_true_graph)
                    metric_f1_edges.update(y_pred_edge, y_true_edge)

                    metric_grad.update(grad_norm)

            timer.end("train")

            train_logs = dict(
                graph_pred_loss=metric_loss_graph_pred,
                edge_pred_loss=metric_loss_edge_pred,
                logits_graph_pos=metric_logits_graph_pos,
                logits_graph_neg=metric_logits_graph_neg,
                logits_edges_pos=metric_logits_edge_pos,
                logits_edges_neg=metric_logits_edge_neg,
                f1_graph=metric_f1_graph,
                f1_edges=metric_f1_edges,
                grad_norm=metric_grad,
                pos_frac_graph=metric_pos_graph,
                pos_frac_edges=metric_pos_edges,
            )

            train_logs = {k: v.compute().cpu().item() for k, v in train_logs.items()}

            tp, fp, tn, fn, sup = metrics_binary.compute().cpu().float().tolist()

            binary_rates = dict(
                tp_graph=tp / (tp + fn),
                fn_graph=fn / (fn + tp),
                fp_graph=fp / (fp + tn),
                tn_graph=tn / (tn + fp),
            )

            train_logs = {**train_logs, **binary_rates}

            metric_f1_graph.reset()
            metric_f1_edges.reset()

            timer.start("eval")
            model.eval()

            test_logs = []

            with torch.no_grad():
                for req_q, resp_q in test_queues:
                    targets = [
                        dataset.get_item()
                        for _ in range(
                            conf.training.rollouts_per_epoch
                            // conf.training.n_test_workers
                        )
                    ]
                    req_q.put({"type": "reset", "targets": targets})
                    req_q.put({"type": "batch"})

                workers_done = [False] * len(test_queues)

                for i in itertools.count():
                    if all(workers_done):
                        break

                    worker_idx = i % len(test_queues)

                    if workers_done[worker_idx]:
                        continue

                    req_q, resp_q = test_queues[worker_idx]

                    response = resp_q.get()

                    if response["type"] == "batch":
                        batch_ids = response.pop("batch_ids")
                        traj_ids = response.pop("traj_ids")

                        targets = (
                            {k: v.to(device) for k, v in response["targets"].items()}
                            if isinstance(response["targets"], dict)
                            else response["targets"].to(device)
                        )

                        subgraphs = {
                            k: v.to(device) for k, v in response["subgraphs"].items()
                        }
                        terminals = response["terminals"].to(device)

                        targets_emb = model.encode_target(targets)[batch_ids]

                        subgraphs_emb = model.encode_query(subgraphs, targets_emb)

                        y_pred_graph, y_pred_mask = model.decode(
                            targets_emb, subgraphs_emb, terminals
                        )

                        req_q.put(
                            {
                                "type": "prediction",
                                "traj_ids": traj_ids,
                                "y_pred_graph": y_pred_graph.reshape(-1).cpu(),
                                "y_pred_mask": y_pred_mask.cpu(),
                            }
                        )
                        req_q.put({"type": "batch"})
                        continue

                    if response["type"] == "done":
                        req_q.put({"type": "summary"})
                        workers_done[worker_idx] = True
                        continue

                    raise RuntimeError(f"unexpected message type {response["type"]}")

            workers_to_plot = []
            for i, (req_q, resp_q) in enumerate(test_queues):
                response = resp_q.get()
                response_type = response["type"]
                assert (
                    response_type == "summary"
                ), f"expected 'summary' but got '{response_type}'"
                response.pop("type")
                test_logs.append(response)

                # inequality selects workers with both correct and incorrect samples
                if 0.0 < response["correct"] < 1.0:
                    workers_to_plot.append(i)

            assert len(test_logs) == len(test_queues)

            if fabric.is_global_zero:
                worker_idx = random.choice(workers_to_plot) if workers_to_plot else 0
                req_q, _ = test_queues[worker_idx]
                req_q.put({"type": "plot"})

            timer.end("eval")

            test_logs = fabric.all_gather(test_logs)

            all_logs = defaultdict(list)

            for l in test_logs:
                for k, v in l.items():
                    all_logs[k].append(v)

            test_logs = {
                k: torch.stack(v).reshape(-1).cpu()
                for k, v in all_logs.items()
                if k not in ("predictions",)
            }

            test_logs = {
                "correct": test_logs["correct"].mean(),
                "length_min_pred": test_logs["length_min_pred"].min(),
                "length_min_true": test_logs["length_min_true"].min(),
                "length_mean_pred": test_logs["length_mean_pred"].mean(),
                "length_mean_true": test_logs["length_mean_true"].mean(),
                "length_max_pred": test_logs["length_max_pred"].max(),
                "length_max_true": test_logs["length_max_true"].max(),
                "top_k": test_logs["top_k"].mean(),
                "successors_avg": test_logs["successors_avg"].mean(),
                "successors_sum": test_logs["successors_sum"].sum(),
                "eval_loss": test_logs["eval_loss"].mean(),
            }
            test_logs = {k: v.item() for k, v in test_logs.items()}

            if not fabric.is_global_zero:
                continue

            torch.save(
                model.module,
                directory / f"{conf.environment.dataset}_{conf.environment.task}.pt",
            )

            total_done = (
                (epoch + 1) * conf.training.samples_per_epoch * fabric.world_size
            )

            meta_logs = dict(
                epoch=epoch,
                step=total_done / 1e6,
                fds_used=proc.num_fds(),
                lr=optimizer.param_groups[0]["lr"],
            )

            time_quantiles = torch.cat(
                [
                    torch.arange(0, 10000, 1000),
                    torch.arange(9000, 10000, 100),
                    torch.tensor([9925, 9950, 9975, 9999, 10000]),
                ]
            )

            time_buckets = torch.quantile(
                torch.cat(time_logs).cpu(), time_quantiles / 10000
            )

            time_logs = dict(zip(time_quantiles.tolist(), time_buckets.tolist()))

            timer.end("epoch")

            timer_logs = timer.compute()
            model_logs = model.param_norm()

            system_logs = sys_monitor.get_metrics()

            logs = {
                **train_logs,
                **meta_logs,
                **test_logs,
                **timer_logs,
                **model_logs,
                **system_logs,
            }
            training_logs.append(logs)

            plot_training(
                training_logs,
                time_logs,
                directory,
                list(conf.overrides),
            )

            logs = {
                k: round(v, 6) if isinstance(v, float) else v for k, v in logs.items()
            }

            logger.info(logs)
    except Exception as e:
        # TODO kill job on exception
        print(e)
        traceback.print_exc()
