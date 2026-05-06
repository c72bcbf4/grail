import itertools
import os
import sys
import time
import uuid
from pathlib import Path

import numpy as np
import pandas as pd
import psutil
import torch
from hydra import initialize, compose
from hydra.utils import instantiate
from loguru import logger
from matplotlib import pyplot as plt
from tqdm import trange

from src.data import GraphDataset
from src.logging import configure_logging
from src.rollout import rollout_test


def plot_count_bars(data):
    keys = list(data.keys())

    values = {k: data[k].detach().cpu().flatten().numpy() for k in keys}

    max_idx = 0
    for v in values.values():
        idx = np.where(~np.isnan(v))[0]
        if len(idx) > 0:
            max_idx = max(max_idx, idx.max())

    steps = np.arange(max_idx + 1)
    n_keys = len(keys)

    width = 1.0
    group_width = n_keys * width
    group_gap = 1.0

    group_centers = steps * (group_width + group_gap)
    offsets = (np.arange(n_keys) - (n_keys - 1) / 2) * width

    fig, ax = plt.subplots(figsize=(18, 5))

    for j, k in enumerate(keys):
        v = values[k][: max_idx + 1]
        mask = ~np.isnan(v)

        ax.bar(group_centers[mask] + offsets[j], v[mask], width, label=k)

    ax.set_xticks(group_centers)
    ax.set_xticklabels(steps, rotation=90)

    ax.set_xlabel("Step")
    ax.set_ylabel("#Successors")
    ax.legend()

    plt.savefig("appendix_successors")
    plt.tight_layout()
    plt.close()


def plot_grouped_times(labels, data):
    keys = list(data.keys())

    n_groups = len(labels)
    n_keys = len(keys)

    x = np.arange(n_groups)

    width = 20.0
    group_gap = 10.0

    group_width = n_keys * width
    group_centers = x * (group_width + group_gap)

    offsets = (np.arange(n_keys) - (n_keys - 1) / 2) * width

    fig, ax = plt.subplots(figsize=(14, 5))

    for i, k in enumerate(keys):
        values = data[k]
        ax.bar(group_centers + offsets[i], values, width, label=k)

    ax.set_xticks(group_centers)
    ax.set_xticklabels(labels, rotation=90)

    ax.set_xlabel("Quantiles")
    ax.set_ylabel("Time (ms)")
    ax.legend()

    plt.tight_layout()
    plt.savefig("appendix_quantiles")


directory_test = Path(f"outputs/test/{uuid.uuid4()}")

n_workers = 32
n_rollouts_total = 1000
n_rollouts_per_batch = 1000
batch_size = 1024


datasets = ["trees", "coloring_15", "coloring_20", "qm9", "pubchem_32", "pubchem_40"]
tasks = ["graph"]

evals = list(itertools.product(datasets, tasks))


all_logs = []
all_times = {}
all_successors = {}

time_quantiles = torch.cat(
    [
        torch.arange(0, 10000, 1000),
        torch.arange(9000, 10000, 100),
        torch.tensor([9925, 9950, 9975, 9999, 10000]),
    ]
)

configure_logging(filename=directory_test / "test.log")
logger.info("starting")

for dataset_name, task_name in evals:
    logger.info(f"evaluating {dataset_name} {task_name}")

    with initialize(version_base=None, config_path="conf"):
        conf = compose(
            config_name="prod_hk",
            overrides=[f"environment={dataset_name}"],
        )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model_path = f"models/{dataset_name}_{task_name}.pt"
    model = torch.load(model_path, weights_only=False).to(device)

    conf.environment.task = task_name
    generator = instantiate(conf.environment, train=False)
    dataset = GraphDataset(generator)

    dataset.get_item()

    mk_q = lambda: torch.multiprocessing.Queue(maxsize=1)
    test_queues = [(mk_q(), mk_q()) for _ in range(n_workers)]

    logger.info("starting workers")
    test_workers = [
        torch.multiprocessing.Process(
            target=rollout_test,
            args=(req_q, resp_q, conf.environment, directory_test, batch_size, True),
            daemon=True,
        )
        for i, (req_q, resp_q) in enumerate(test_queues)
    ]

    for w in test_workers:
        w.start()

    logger.info("starting workers done")

    ids_seen = set()

    test_logs = []
    time_logs = []
    succ_logs = []

    start = time.time()

    with torch.inference_mode(), torch.autocast(device_type="cuda"):
        logger.info("starting rollouts")

        for _ in trange(
            n_rollouts_total // n_rollouts_per_batch, file=sys.stdout, leave=False
        ):
            for req_q, resp_q in test_queues:
                targets = []
                while len(targets) < n_rollouts_per_batch // n_workers:
                    sample, info = dataset.get_item(info=True)

                    if "id" in info:
                        if info["id"] in ids_seen:
                            continue
                        ids_seen.add(info["id"])
                    targets.append(sample)

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

                    time_logs.append(response["times"])

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

            for i, (req_q, resp_q) in enumerate(test_queues):
                response = resp_q.get()
                response_type = response["type"]
                assert (
                    response_type == "summary"
                ), f"expected 'summary' but got '{response_type}'"
                response.pop("type")
                test_logs.append(response)

            pid = os.getpid()
            proc = psutil.Process(pid)
            num_fds = proc.num_fds()

            print(f"num_fds: {num_fds}")

        end = time.time()

        successors = torch.cat([l.pop("step_successors") for l in test_logs])

        mask = successors != -1
        successors_avg = (successors * mask).sum(dim=0) / mask.sum(dim=0)

        plt.bar(range(successors_avg.shape[0]), successors_avg)
        plt.xlabel("Step")
        plt.ylabel("Successors")
        plt.show()

        successors_flat = successors[successors != -1]

        test_logs = pd.DataFrame(test_logs)

        correct = test_logs["correct"].mean().item()
        test_lengths_pred = test_logs["length_mean_pred"].mean().item()
        test_lengths_true = test_logs["length_mean_true"].mean().item()
        duration = end - start
        duration_per_sample = duration / n_rollouts_total

        logs = {
            "dataset": dataset_name,
            "task": task_name,
            "correct": correct,
            "duration": duration,
            "duration_per_sample": duration_per_sample,
            "successor_avg": successors_flat.mean().item(),
            "len_avg_pred": test_lengths_pred,
            "len_avg_true": test_lengths_true,
        }

        logs = {
            k: round(v, 6) if not isinstance(v, str) else v for k, v in logs.items()
        }

        dataset_name = dataset_name.replace("_", " ").upper()

        all_logs.append(logs)
        all_successors[dataset_name] = successors_avg

        all_times[dataset_name] = torch.quantile(
            torch.cat(time_logs), time_quantiles / 10000
        )

        for w in test_workers:
            w.kill()

        print("\n".join(map(str, all_logs)))

plot_grouped_times([x / 100 for x in time_quantiles.tolist()], all_times)
plot_count_bars(all_successors)


# {'dataset': 'trees', 'task': 'graph', 'correct': 0.998, 'duration': 8.178046, 'duration_per_sample': 0.008178, 'successor_avg': 8.99824, 'len_avg_pred': 9.930001, 'len_avg_true': 9.944001}
# {'dataset': 'trees', 'task': 'image', 'correct': 0.977, 'duration': 25.127519, 'duration_per_sample': 0.025128, 'successor_avg': 9.119061, 'len_avg_pred': 9.934999, 'len_avg_true': 10.038}
# {'dataset': 'trees', 'task': 'fingerprint', 'correct': 0.636, 'duration': 6.4261, 'duration_per_sample': 0.006426, 'successor_avg': 12.920767, 'len_avg_pred': 7.958, 'len_avg_true': 9.868999}

# {'dataset': 'coloring_15', 'task': 'graph', 'correct': 0.94, 'duration': 18.842516, 'duration_per_sample': 0.018843, 'successor_avg': 20.92029, 'len_avg_pred': 19.892001, 'len_avg_true': 20.715}
# {'dataset': 'coloring_15', 'task': 'image', 'correct': 0.686, 'duration': 33.120237, 'duration_per_sample': 0.03312, 'successor_avg': 22.809324, 'len_avg_pred': 18.312003, 'len_avg_true': 21.533999}
# {'dataset': 'coloring_15', 'task': 'fingerprint', 'correct': 0.682, 'duration': 14.73305, 'duration_per_sample': 0.014733, 'successor_avg': 23.28233, 'len_avg_pred': 17.626002, 'len_avg_true': 21.173999}

# {'dataset': 'coloring_20', 'task': 'graph', 'correct': 0.873, 'duration': 27.521265, 'duration_per_sample': 0.027521, 'successor_avg': 30.491089, 'len_avg_pred': 23.92, 'len_avg_true': 26.463003}
# {'dataset': 'coloring_20', 'task': 'image', 'correct': 0.561, 'duration': 38.915281, 'duration_per_sample': 0.038915, 'successor_avg': 32.856583, 'len_avg_pred': 19.720001, 'len_avg_true': 26.917}
# {'dataset': 'coloring_20', 'task': 'fingerprint', 'correct': 0.601, 'duration': 19.792298, 'duration_per_sample': 0.019792, 'successor_avg': 30.687929, 'len_avg_pred': 19.447001, 'len_avg_true': 25.602996}

# {'dataset': 'qm9', 'task': 'graph', 'correct': 1.0, 'duration': 7.678542, 'duration_per_sample': 0.007679, 'successor_avg': 12.221186, 'len_avg_pred': 10.433003, 'len_avg_true': 10.433003}
# {'dataset': 'qm9', 'task': 'image', 'correct': 1.0, 'duration': 17.469873, 'duration_per_sample': 0.01747, 'successor_avg': 11.713277, 'len_avg_pred': 10.458, 'len_avg_true': 10.458}
# {'dataset': 'qm9', 'task': 'fingerprint', 'correct': 1.0, 'duration': 6.071778, 'duration_per_sample': 0.006072, 'successor_avg': 11.918461, 'len_avg_pred': 10.399, 'len_avg_true': 10.399}

# {'dataset': 'pubchem_32', 'task': 'graph', 'correct': 0.908, 'duration': 35.353836, 'duration_per_sample': 0.035354, 'successor_avg': 44.452099, 'len_avg_pred': 23.175999, 'len_avg_true': 24.386003}
# {'dataset': 'pubchem_32', 'task': 'image', 'correct': 0.353, 'duration': 51.823005, 'duration_per_sample': 0.051823, 'successor_avg': 45.370762, 'len_avg_pred': 17.576001, 'len_avg_true': 24.983999}
# {'dataset': 'pubchem_32', 'task': 'fingerprint', 'correct': 0.309, 'duration': 33.878432, 'duration_per_sample': 0.033878, 'successor_avg': 62.67992, 'len_avg_pred': 15.705, 'len_avg_true': 24.942}

# {'dataset': 'pubchem_40', 'task': 'graph', 'correct': 0.631, 'duration': 71.183855, 'duration_per_sample': 0.071184, 'successor_avg': 70.638145, 'len_avg_pred': 30.284001, 'len_avg_true': 37.495007}
# {'dataset': 'pubchem_40', 'task': 'image', 'correct': 0.049, 'duration': 63.812309, 'duration_per_sample': 0.063812, 'successor_avg': 63.429184, 'len_avg_pred': 17.367, 'len_avg_true': 37.585002}
# {'dataset': 'pubchem_40', 'task': 'fingerprint', 'correct': 0.129, 'duration': 53.721961, 'duration_per_sample': 0.053722, 'successor_avg': 79.440498, 'len_avg_pred': 16.821998, 'len_avg_true': 37.715999}
