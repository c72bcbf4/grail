from dataclasses import dataclass
from typing import List, TypedDict

from torch import Tensor


class Data(TypedDict):
    nodes: Tensor
    degrees: Tensor
    edges: Tensor
    edge_index: Tensor
    batch_nodes: Tensor
    batch_edges: Tensor
    terminals: Tensor
    extras: Tensor


class Graph(TypedDict):
    nodes: Tensor
    degrees: Tensor
    edges: Tensor
    edges_f: Tensor
    terminal: Tensor
    extra: Tensor


@dataclass
class EnvironmentConfig:
    dataset: str
    min_size: int
    max_size: int
    image_size: int
    n_node_types: int = None
    n_edge_types: int = None
    task: str = None


@dataclass
class GraphEncoderConfig:
    node_embed_dim: int
    edge_embed_dim: int

    hidden_channels: int
    hidden_scale: int

    output_channels: int | None

    num_layers: int
    mode: str
    norm: str

    task: str


class ImageEncoderConfig:
    name: str
    task: str


class FingerprintEncoderConfig:
    embedding_size: int
    dropout: float
    task: str


@dataclass
class DecoderConfig:
    graph: List[int]
    edges: List[int]
    norm: str
    dropout: float


@dataclass
class TrainingConfig:
    samples_per_run: int
    samples_per_epoch: int
    rollouts_per_epoch: int

    total_batch_size: int

    buffer_batch_size: int
    buffer_min_size: int
    buffer_max_size: int
    buffer_multiplier: int

    n_train_workers: int
    n_test_workers: int


@dataclass
class OptimizerConf:
    lr_start: float
    lr_max: float
    lr_end: float
    lr_warmup_samples: int
    lr_total_samples: int

    weight_decay: float

    beta_1: float
    beta_2: float
    eps: float

    gamma: float
    label_smoothing: float


@dataclass
class AppConfig:
    batch: int
    batch_size: int
    batch_idx: int
    output_dir: str
    overrides: list

    environment: EnvironmentConfig
    training: TrainingConfig
    optimizer: OptimizerConf

    encoder_query: GraphEncoderConfig
    decoder: DecoderConfig
    encoder_target: (
        GraphEncoderConfig | ImageEncoderConfig | FingerprintEncoderConfig
    ) = None
