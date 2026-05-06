import hashlib
import math
import os
import queue
import subprocess
from itertools import islice
from threading import Thread
from typing import Any, Iterable, Counter

import networkx as nx
import numpy as np
import psutil
import rustworkx as rx
import torch
import torch_geometric
import torchvision
from pynvml import (
    nvmlShutdown,
    nvmlDeviceGetHandleByIndex,
    nvmlInit,
    nvmlDeviceGetUtilizationRates,
)
from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator
from torch import Tensor, nn
from torch.optim.lr_scheduler import _LRScheduler
from torch_geometric.nn import GraphNorm
from torchmetrics import MeanMetric, CatMetric

from src.definitions import Data, Graph


class SystemMonitor:
    def __init__(self, interval: float = 0.25):
        self.interval = interval

        args = dict(sync_on_compute=False)
        self.gpu_util = CatMetric(**args)
        self.cpu_util = CatMetric(**args)

        self.gpu_mem = CatMetric(**args)
        self.cpu_mem = CatMetric(**args)

        self.worker = Thread(target=self.get_utilization, daemon=True)
        self.collect = True

    def start(self):
        self.worker.start()
        return self

    def stop(self):
        self.collect = False

    @torch.no_grad()
    def get_utilization(self):
        nvmlInit()
        handle = nvmlDeviceGetHandleByIndex(0)

        while self.collect:
            if handle is not None:
                utilization = nvmlDeviceGetUtilizationRates(handle)

                self.gpu_util.update(utilization.gpu)
                self.gpu_mem.update(utilization.memory)

            # cpu_percent blocks for interval, no need for sleep
            cpu_util = psutil.cpu_percent(self.interval, percpu=False)
            cpu_ram = psutil.virtual_memory().percent

            self.cpu_mem.update(cpu_ram)
            self.cpu_util.update(cpu_util)

        nvmlShutdown()

    @torch.no_grad()
    def get_metrics(self):
        gpu_util = self.gpu_util.compute()
        cpu_util = self.cpu_util.compute()
        gpu_mem = self.gpu_mem.compute()
        cpu_mem = self.cpu_mem.compute()

        stats = dict(
            sys_gpu_util_mean=gpu_util.mean().item() if len(gpu_util) != 0 else 0.0,
            sys_gpu_util_max=gpu_util.max().item() if len(gpu_util) != 0 else 0.0,
            sys_cpu_util_mean=cpu_util.mean().item() if len(cpu_util) != 0 else 0.0,
            sys_cpu_util_max=cpu_util.max().item() if len(cpu_util) != 0 else 0.0,
            sys_gpu_mem_mean=gpu_mem.mean().item() if len(gpu_mem) != 0 else 0.0,
            sys_gpu_mem_max=gpu_mem.max().item() if len(gpu_mem) != 0 else 0.0,
            sys_cpu_mem_mean=cpu_mem.mean().item() if len(cpu_mem) != 0 else 0.0,
            sys_cpu_mem_max=cpu_mem.max().item() if len(cpu_mem) != 0 else 0.0,
        )

        for m in [self.gpu_util, self.cpu_util, self.gpu_mem, self.cpu_mem]:
            m.reset()

        return stats


class Batcher:
    def __init__(self, queues, device):
        self.train_worker_idx = -1

        self.queues = queues
        self.device = device

    def get(self):
        while True:
            try:
                self.train_worker_idx += 1
                in_q, out_q = self.queues[self.train_worker_idx % len(self.queues)]
                response = out_q.get_nowait()
            except queue.Empty:
                continue

            targets = response.pop("targets")
            subgraphs = response.pop("subgraphs")

            response = {k: v.to(self.device) for k, v in response.items()}
            targets = (
                {k: v.to(self.device) for k, v in targets.items()}
                if isinstance(targets, dict)
                else targets.to(self.device)
            )
            subgraphs = {k: v.to(self.device) for k, v in subgraphs.items()}

            response["targets"] = targets
            response["subgraphs"] = subgraphs

            return response, in_q


class EWMA:
    def __init__(self, decay: float = 0.9):
        self.decay = decay
        self.ema = None

    def update(self, value: float | Tensor):
        if self.ema is None:
            self.ema = value
        else:
            self.ema = self.decay * self.ema + (1 - self.decay) * value

    def compute(self) -> float:
        return self.ema

    def reset(self):
        self.ema = None


class CosineWarmupDecayLR(_LRScheduler):
    def __init__(
        self,
        optimizer,
        lr_start,
        lr_max,
        lr_end,
        total_steps,
        warmup_steps,
        last_epoch=-1,
    ):
        self.lr_start = lr_start
        self.lr_max = lr_max
        self.lr_end = lr_end
        self.total_steps = total_steps
        self.warmup_steps = warmup_steps
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        step = self.last_epoch + 1

        if step <= self.warmup_steps:
            return [
                self.lr_start
                + 0.5
                * (self.lr_max - self.lr_start)
                * (1 - math.cos(math.pi * step / self.warmup_steps))
                for _ in self.optimizer.param_groups
            ]

        if step <= self.total_steps:
            decay_step = step - self.warmup_steps
            decay_total = self.total_steps - self.warmup_steps
            return [
                self.lr_end
                + 0.5
                * (self.lr_max - self.lr_end)
                * (1 + math.cos(math.pi * decay_step / decay_total))
                for _ in self.optimizer.param_groups
            ]

        return [self.lr_end for _ in self.optimizer.param_groups]


class AutoResetMeanMetric(MeanMetric):
    def __init__(self, **kwargs: Any):
        super().__init__(**kwargs)

    def compute(self) -> Tensor:
        value = super().compute()
        self.reset()
        return value


@torch.no_grad()
def batch_fast(data: Iterable[Graph], device: str = None):
    nodes, degrees, edges, edges_f, terminals, extras, batch_nodes, edge_sizes = tuple(
        [] for _ in range(8)
    )

    # count manually as data may be stream
    batch_size = 0

    for x in data:
        n = x["nodes"]
        d = x["degrees"]
        e = x["edges"]
        f = x["edges_f"]
        t = x["terminal"]
        ex = x["extra"]
        n_nodes = n.shape[0]
        n_edges = e.shape[1]

        nodes.append(n)
        degrees.append(d)
        edges.append(e)
        edges_f.append(f)
        terminals.append(t)
        extras.append(ex)

        batch_nodes.append(n_nodes)
        edge_sizes.append(n_edges)

        batch_size += 1

    # is used for both edge offsets and batch index
    batch_nodes = torch.as_tensor([0] + batch_nodes)

    edge_sizes = torch.tensor(edge_sizes)

    nodes_batch = torch.cat(nodes, dim=0)
    degrees_batch = torch.cat(degrees, dim=0)
    edges_f_batch = torch.cat(edges_f, dim=0)
    edges_batch = torch.cat(edges, dim=1)

    terminals_batch = torch.as_tensor(terminals).float().unsqueeze(-1)
    extras = torch.cat(extras, dim=0)

    edge_offsets = torch.as_tensor(batch_nodes[:-1]).cumsum(0)
    edge_offsets = edge_offsets.repeat_interleave(edge_sizes)
    edge_offsets = edge_offsets.to(edges_batch.device)

    edge_index = edges_batch + edge_offsets

    arange = torch.arange(batch_size)

    batch_nodes = arange.repeat_interleave(batch_nodes[1:])
    batch_edges = arange.repeat_interleave(edge_sizes)

    if device is not None:
        nodes_batch = nodes_batch.to(device, non_blocking=True)
        degrees_batch = degrees_batch.to(device, non_blocking=True)
        edges_f_batch = edges_f_batch.to(device, non_blocking=True)
        edge_index = edge_index.to(device, non_blocking=True)
        batch_nodes = batch_nodes.to(device, non_blocking=True)
        batch_edges = batch_edges.to(device, non_blocking=True)
        terminals_batch = terminals_batch.to(device, non_blocking=True)
        extras = extras.to(device, non_blocking=True)

    return Data(
        nodes=nodes_batch,
        degrees=degrees_batch,
        edges=edges_f_batch,
        edge_index=edge_index,
        terminals=terminals_batch,
        batch_nodes=batch_nodes,
        batch_edges=batch_edges,
        extras=extras,
    )


def make_norm1d(norm, channels):
    if norm is None:
        return torch.nn.Identity()

    if norm == "LayerNorm":
        return nn.LayerNorm(channels)

    if norm == "InstanceNorm":
        return torch.nn.InstanceNorm1d(channels)

    if norm == "GraphNorm":
        return GraphNorm(channels)

    raise ValueError(f"unknown normalization layer '{norm}'!")


def count_parameters(model):
    if model is None:
        return -1

    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def batch_stream(iterable, n):
    it = iter(iterable)
    while True:
        batch = list(islice(it, n))
        if not batch:
            break
        yield batch


def patch_environment():
    if os.environ.get("SYSTEMNAME", "") not in [
        "juwelsbooster",
        "juwels",
        "jurecadc",
        "jusuf",
    ]:
        print("not patching")
        return

    if "SLURM_JOB_NODELIST" not in os.environ:
        return

    nodelist = os.environ["SLURM_JOB_NODELIST"]
    hostnames = (
        subprocess.check_output(["scontrol", "show", "hostnames", nodelist])
        .decode()
        .splitlines()
    )
    master_addr = hostnames[0]

    master_addr = master_addr + "i"
    print(f"patching MASTER_ADDR={master_addr}")
    os.environ["MASTER_ADDR"] = master_addr


def node_match(t_data, q_data):
    if q_data.get("type") != t_data.get("type"):
        return False

    return True


def edge_match(e1, e2):
    return e1.get("type") == e2.get("type")


def is_subgraph_exact(query: rx.PyGraph, target: rx.PyGraph):
    return rx.is_subgraph_isomorphic(
        target,
        query,
        node_matcher=node_match,
        edge_matcher=edge_match,
        id_order=False,
        induced=False,
    )


def label_smoothing(y: torch.Tensor, eps: float) -> torch.Tensor:
    return y * (1 - eps) + (1 - y) * eps


def cycle_count_features(graph, n_max):
    N = graph.num_nodes()
    features = torch.zeros((N, n_max), dtype=torch.float32)

    A = torch.from_numpy(rx.adjacency_matrix(graph)).float()
    A_power = A.clone()

    for n in range(1, n_max):
        A_power @= A
        features[:, n - 1] = torch.diag(A_power)

    return features.log1p()[:, 1:]


def non_backtracking_cycle_features(graph, k_max):
    edges = list(graph.edge_list())
    num_nodes = graph.num_nodes()

    dir_edges = []
    for u, v in edges:
        dir_edges.append((u, v))
        dir_edges.append((v, u))

    edge_index = {e: i for i, e in enumerate(dir_edges)}
    M = len(dir_edges)

    B = torch.zeros((M, M), dtype=torch.float32)

    for i, (u, v) in enumerate(dir_edges):
        for w in graph.neighbors(v):
            if w != u:
                j = edge_index[(v, w)]
                B[i, j] = 1.0

    Bk = B.clone()
    edge_feats = []

    for _ in range(k_max):
        edge_feats.append(torch.diag(Bk))
        Bk = Bk @ B

    edge_feats = torch.stack(edge_feats, dim=1)

    node_feats = torch.zeros((num_nodes, k_max), dtype=torch.float32)
    for (u, _), i in edge_index.items():
        node_feats[u] += edge_feats[i]

    return node_feats.log1p()


def laplacian_pe(g, k):
    n = g.num_nodes()

    A = torch.from_numpy(rx.adjacency_matrix(g)).to(dtype=torch.float)
    deg = A.sum(dim=1)

    d_inv_sqrt = deg.clamp(min=1e-8).pow(-0.5)
    D_inv_sqrt = torch.diag(d_inv_sqrt)
    L = torch.eye(n, device=A.device) - D_inv_sqrt @ A @ D_inv_sqrt

    eigvals, eigvecs = torch.linalg.eigh(L)

    eps = 1e-5
    non_zero_mask = eigvals > eps
    pe = eigvecs[:, non_zero_mask]

    if pe.size(1) > 0:
        max_abs_idx = torch.argmax(torch.abs(pe), dim=0)
        signs = torch.sign(pe[max_abs_idx, torch.arange(pe.shape[1])])
        pe = pe * signs

    if pe.size(1) > k:
        pe = pe[:, :k]
    elif pe.size(1) < k:
        pe = torch.nn.functional.pad(pe, (0, k - pe.size(1)), value=0.0)

    return pe


def configure_weight_decay(model, weight_decay):
    decay = set()
    no_decay = set()
    whitelist_weight_modules = (
        torch.nn.Linear,
        torch.nn.MultiheadAttention,
        torch.nn.modules.conv.Conv2d,
        torch_geometric.nn.dense.linear.Linear,
        torchvision.models.convnext.CNBlock,
        torchvision.models.swin_transformer.ShiftedWindowAttention,
        torchvision.models.swin_transformer.ShiftedWindowAttentionV2,
    )
    blacklist_weight_modules = (
        torch.nn.LayerNorm,
        torch.nn.BatchNorm1d,
        torch.nn.BatchNorm2d,
        torch.nn.Embedding,
    )

    global_skip = ("embeddings_fingerprint",)

    for mn, m in model.named_modules():
        # Only inspect parameters owned by this module (avoid double-counting via recursion).
        for pn, p in m.named_parameters(recurse=False):
            fpn = "%s.%s" % (mn, pn) if mn else pn  # full param name

            if pn in global_skip:
                no_decay.add(fpn)
                continue

            if pn.endswith("bias"):
                # all biases will not be decayed
                no_decay.add(fpn)
                continue

            # Scalars / vectors (e.g., GIN/GINE trainable eps, norm gains) typically should not be decayed.
            if p.ndim <= 1:
                no_decay.add(fpn)
                continue

            # fmt: off
            is_weight = any(
                (
                    pn.endswith("weight"),
                    pn.endswith("layer_scale"),  # from ConvNeXt
                    pn.endswith("logit_scale"),  # from Swin v2 Transformer
                )
            )
            # fmt: on

            if is_weight and isinstance(m, whitelist_weight_modules):
                # weights of whitelist modules will be weight decayed
                decay.add(fpn)
                continue

            if is_weight and isinstance(m, blacklist_weight_modules):
                # weights of blacklist modules will NOT be weight decayed
                no_decay.add(fpn)
                continue

            raise ValueError(
                f"unexpected parameter {fpn} to be decayed for module {type(m)}!"
            )

    # validate that we considered every parameter
    param_dict = {pn: p for pn, p in model.named_parameters()}

    # special-case: only add pos_emb if it actually exists
    if "pos_emb" in param_dict:
        no_decay.add("pos_emb")

    inter_params = decay & no_decay
    union_params = decay | no_decay
    assert (
        len(inter_params) == 0
    ), "parameters %s made it into both decay/no_decay sets!" % (str(inter_params),)
    assert (
        len(param_dict.keys() - union_params) == 0
    ), "parameters %s were not separated into either decay/no_decay set!" % (
        str(param_dict.keys() - union_params),
    )

    # create the pytorch optimizer object
    optim_groups = [
        {
            "params": [param_dict[pn] for pn in sorted(list(decay))],
            "weight_decay": weight_decay,
        },
        {
            "params": [param_dict[pn] for pn in sorted(list(no_decay))],
            "weight_decay": 0.0,
        },
    ]
    return optim_groups


def can_be_valid_subgraph(query: rx.PyGraph, target: rx.PyGraph) -> bool:
    query_node_types = [data["type"] for data in query.nodes()]
    target_node_types = [data["type"] for data in target.nodes()]
    query_edge_types = [data["type"] for data in query.edges()]
    target_edge_types = [data["type"] for data in target.edges()]

    query_node_count = Counter(query_node_types)
    target_node_count = Counter(target_node_types)
    query_edge_count = Counter(query_edge_types)
    target_edge_count = Counter(target_edge_types)

    for t, count in query_node_count.items():
        if count > target_node_count.get(t, 0):
            return False
    for t, count in query_edge_count.items():
        if count > target_edge_count.get(t, 0):
            return False
    return True


def get_mol_fingerprint(smiles, size: int = 2048, radius=2) -> np.ndarray:
    mol = Chem.MolFromSmiles(smiles)
    fpgen = rdFingerprintGenerator.GetMorganGenerator(fpSize=size, radius=radius)
    return fpgen.GetFingerprintAsNumPy(mol)


def get_graph_fingerprint(G: nx.Graph, size: int = 2048, radius=2) -> np.ndarray:
    bitvector = np.zeros(size, dtype=np.uint8)

    node_hashes = {}
    for n in G.nodes():
        node_type = str(G.nodes[n].get("type", "unknown"))
        label = f"type:{node_type}"
        node_hashes[n] = int(hashlib.md5(label.encode()).hexdigest(), 16)

        bitvector[node_hashes[n] % size] = 1

    for r in range(1, radius + 1):
        new_hashes = {}
        for n in G.nodes():
            neighbors_info = []
            for nbr in G.neighbors(n):
                edge_type = str(G[n][nbr].get("type", "default"))
                neighbors_info.append((edge_type, node_hashes[nbr]))

            neighbors_info.sort()

            combined = f"{node_hashes[n]}_" + "_".join(
                [f"{e}:{h}" for e, h in neighbors_info]
            )
            new_hash = int(hashlib.md5(combined.encode()).hexdigest(), 16)
            new_hashes[n] = new_hash

            bitvector[new_hash % size] = 1

        node_hashes = new_hashes

    return bitvector
