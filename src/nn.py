from typing import Tuple

import torch
import torchvision.models
from hydra.utils import instantiate
from loguru import logger
from torch import nn
from torch.nn import ModuleList, Linear
from torch.nn.utils import parameters_to_vector
from torch_geometric.nn import (
    JumpingKnowledge,
    GINEConv,
    MultiAggregation,
)

from src.data import GraphDataset
from src.definitions import (
    Data,
    GraphEncoderConfig,
    DecoderConfig,
    ImageEncoderConfig,
    FingerprintEncoderConfig,
)
from src.util import make_norm1d, batch_fast


class MultiLayerPerceptron(torch.nn.Module):
    def __init__(
        self,
        input_size: int,
        output_size: int,
        hidden_channels: list,
        hidden_activation: nn.Module = None,
        bias: bool = True,
        norm: str = None,
        dropout: float = None,
        output_activation: nn.Module = None,
    ):
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.hidden_channels = hidden_channels
        self.bias = bias
        self.norm = norm
        self.dropout = dropout
        self.hidden_activation = hidden_activation or nn.ReLU()

        in_sizes = [input_size] + hidden_channels
        out_sizes = hidden_channels + [output_size]

        hidden_layers = []
        for c_in, c_out in zip(in_sizes[:-1], out_sizes[:-1]):
            linear = nn.Linear(c_in, c_out, bias=bias)
            norm = make_norm1d(self.norm, c_out)

            hidden_layers.append(linear)
            hidden_layers.append(norm)
            hidden_layers.append(self.hidden_activation)

            if self.dropout is not None:
                dropout = nn.Dropout(self.dropout)
                hidden_layers.append(dropout)

        linear = nn.Linear(in_sizes[-1], out_sizes[-1])

        hidden_layers.append(linear)

        if output_activation is not None:
            hidden_layers.append(output_activation)

        self.module = nn.Sequential(*hidden_layers)

    @property
    def in_channels(self):
        return self.input_size

    @property
    def n_params(self):
        return sum(p.numel() for p in self.module.parameters())

    def forward(self, x):
        return self.module(x)


class GINE(nn.Module):
    def __init__(
        self,
        node_embed_dim: int,
        edge_embed_dim: int,
        extra_dim: int,
        hidden_channels: int,
        hidden_scale: int,
        output_channels: int | None,
        num_layers: int,
        bias: bool,
        act: nn.Module,
        mode: str,
        norm: str,
        film_dim: int | None,
    ):
        super().__init__()

        self.node_embed_dim = node_embed_dim
        self.extra_dim = extra_dim
        self.edge_embed_dim = edge_embed_dim

        self.hidden_channels = hidden_channels
        self.hidden_scale = hidden_scale
        self.output_channels = output_channels
        self.bias = bias
        self.act = act
        self.mode = mode
        self.norm = norm

        self.film_dim = film_dim

        self.node_convs = ModuleList()
        self.node_norms = ModuleList()
        self.film_layers = ModuleList()

        self.final_dims = {
            "last": hidden_channels,
            "cat": hidden_channels * num_layers,
        }

        assert self.mode in self.final_dims

        self.linear_nodes = nn.Linear(node_embed_dim + extra_dim, hidden_channels)
        self.linear_edges = nn.Linear(edge_embed_dim, hidden_channels)

        for i in range(num_layers):
            self.node_convs.append(
                # regular GINEConv uses ReLU, it seems important to keep it like that.
                # the rationale is that it transforms individual messages and keeps the positive part
                # only, otherwise features could cancel. this way, the summation is stable.
                GINEConv(
                    nn.Sequential(
                        nn.Linear(hidden_channels, hidden_scale * hidden_channels),
                        self.act,
                        nn.Linear(hidden_scale * hidden_channels, hidden_channels),
                    ),
                    edge_dim=None,
                    train_eps=True,
                )
            )

            self.node_norms.append(make_norm1d(norm, hidden_channels))

            self.film_layers.append(
                nn.Linear(self.film_dim, 2 * hidden_channels, bias=bias)
                if self.film_dim is not None
                else None
            )

        self.jk_layer_node = (
            JumpingKnowledge(mode, channels=hidden_channels, num_layers=num_layers)
            if mode != "last"
            else None
        )

        final_dim = self.final_dims[self.mode]
        self.final_lin_nodes = (
            Linear(final_dim, self.output_channels, bias=bias)
            if self.output_channels is not None
            else None
        )

        self.graph_embedding_size = (
            hidden_channels if mode != "cat" else hidden_channels * num_layers
        )

    def forward(self, nodes, edge_index, edge_attr, batch, conditioning=None):
        node_jks = []

        nodes = self.linear_nodes(nodes)
        edge_attr = self.linear_edges(edge_attr)

        for conv, norm_node in zip(self.node_convs, self.node_norms):
            nodes_i = conv(nodes, edge_index, edge_attr=edge_attr)
            nodes_i = norm_node(nodes_i)
            nodes_i = self.act(nodes_i)

            nodes = nodes + nodes_i

            node_jks.append(nodes)

        nodes = (
            self.jk_layer_node(node_jks)
            if self.jk_layer_node is not None
            else node_jks[-1]
        )

        if self.final_lin_nodes is not None:
            nodes = self.final_lin_nodes(nodes)

        return nodes


class GraphEncoder(nn.Module):
    def __init__(
        self,
        conf: GraphEncoderConfig,
        node_dim: int,
        edge_dim: int,
        extra_dim: int,
        hidden_activation: nn.Module,
        film_dim: int | None,
    ):
        super().__init__()
        self.node_dim = node_dim
        self.edge_dim = edge_dim

        self.node_embed_dim = conf.node_embed_dim
        self.edge_embed_dim = conf.edge_embed_dim

        self.extra_dim = extra_dim
        self.hidden_activation = hidden_activation

        self.film_dim = film_dim

        self.node_embedding = torch.nn.Embedding(self.node_dim, self.node_embed_dim)
        self.edge_embedding = torch.nn.Embedding(self.edge_dim, self.edge_embed_dim)

        ctor = instantiate(
            conf,
            extra_dim=extra_dim,
            act=self.hidden_activation,
            bias=True,
            film_dim=self.film_dim,
        )
        del ctor.keywords["task"]
        self.gnn = ctor()

        # move global pooling into gnns themselves?
        aggrs = ["SumAggregation"]
        self.graph_pool = MultiAggregation(aggrs)
        self.graph_embedding_size = len(aggrs) * self.gnn.graph_embedding_size

    def forward(self, batch_x: Data, conditioning=None):
        nodes = batch_x["nodes"]
        edge_attr = batch_x["edges"]
        edge_index = batch_x["edge_index"]
        extras = batch_x["extras"]
        batch_nodes = batch_x["batch_nodes"]

        nodes = self.node_embedding(nodes)
        nodes = torch.cat([nodes, extras], dim=-1)
        edge_attr = self.edge_embedding(edge_attr)

        node_embedding = self.gnn(
            nodes=nodes,
            edge_index=edge_index,
            edge_attr=edge_attr,
            batch=batch_nodes,
            conditioning=conditioning,
        )

        graph_embedding = self.graph_pool(node_embedding, batch_nodes)

        return graph_embedding


class ImageEncoder(nn.Module):
    def __init__(self, conf: ImageEncoderConfig):
        super().__init__()
        self.backbone_name = conf.name

        factories = {
            "mobilenet_v3_small": lambda: torchvision.models.mobilenet_v3_small(
                weights=torchvision.models.MobileNet_V3_Small_Weights.DEFAULT
            ),
            "mobilenet_v3_large": lambda: torchvision.models.mobilenet_v3_large(
                weights=torchvision.models.MobileNet_V3_Large_Weights.DEFAULT
            ),
            "resnet": lambda: torchvision.models.resnet50(
                weights=torchvision.models.ResNet50_Weights.DEFAULT
            ),
        }
        self.backbone = factories[self.backbone_name]()

        if self.backbone_name in ("mobilenet_v3_small", "mobilenet_v3_large"):
            assert hasattr(self.backbone, "classifier")
            self.backbone.classifier = nn.Identity()

        if self.backbone_name == "resnet":
            assert hasattr(self.backbone, "fc")
            self.backbone.fc = nn.Identity()

    def forward(self, inputs):
        return self.backbone(inputs)


class FingerprintEncoder(nn.Module):
    def __init__(
        self,
        conf: FingerprintEncoderConfig,
        sample: dict,
        activation: nn.Module,
    ):
        super().__init__()
        self.finger_print_input_size = int(sample["fingerprint"].shape[-1])
        self.finger_print_output_size = conf.embedding_size

        self.dropout = conf.dropout

        self.embeddings_fingerprint = nn.Parameter(
            torch.empty(self.finger_print_input_size, self.finger_print_output_size)
        )
        torch.nn.init.xavier_uniform_(self.embeddings_fingerprint)

        self.encoder_fingerprint = MultiLayerPerceptron(
            input_size=self.finger_print_output_size,
            hidden_channels=[2 * self.finger_print_output_size] * 2,
            hidden_activation=activation,
            output_size=self.finger_print_output_size,
            norm="LayerNorm",
            dropout=self.dropout,
        )

    def forward(self, inputs):
        inputs = inputs.float() @ self.embeddings_fingerprint
        return self.encoder_fingerprint(inputs)


class LatentToGraphDecoder(nn.Module):
    def __init__(
        self,
        dataset: GraphDataset,
        encoder_target: (
            GraphEncoderConfig | ImageEncoderConfig | FingerprintEncoderConfig
        ),
        encoder_query: GraphEncoderConfig,
        decoder: DecoderConfig,
        activation: nn.Module,
    ):
        super().__init__()
        self.task = dataset.generator.task

        self.activation = activation

        graph = dataset.get_item()
        sample = dataset.get_single_sample(graph, graph)

        self.encoder_target, self.target_embedding_size = self.make_target_encoder(
            encoder_target, sample, dataset
        )

        self.encoder_query = GraphEncoder(
            conf=encoder_query,
            node_dim=len(dataset.generator.node_types),
            edge_dim=len(dataset.generator.edge_types),
            extra_dim=dataset.generator.max_size,
            hidden_activation=self.activation,
            film_dim=None,
        )

        graph_embedding = self.encoder_query(batch_fast([sample["graph"]]))
        self.query_embedding_size = int(graph_embedding.shape[-1])
        logger.info(f"graph embedding size: {self.query_embedding_size}")

        if self.task == "graph":
            self.encoder_target = self.encoder_query
            self.target_embedding_size = self.query_embedding_size

        self.decoder_graph = MultiLayerPerceptron(
            input_size=self.target_embedding_size + self.query_embedding_size + 1,
            hidden_channels=decoder.graph,
            output_size=1,
            hidden_activation=activation,
            dropout=decoder.dropout,
            norm=decoder.norm,
        )

        self.decoder_edge = MultiLayerPerceptron(
            input_size=self.target_embedding_size + self.query_embedding_size,
            hidden_channels=decoder.edges,
            output_size=len(dataset.mask),
            hidden_activation=activation,
            dropout=decoder.dropout,
            norm=decoder.norm,
        )

    def _forward_unimplemented(self, *input):
        pass

    def make_target_encoder(self, conf, sample, dataset) -> Tuple[nn.Module, int]:
        if self.task == "graph":
            encoder = GraphEncoder(
                conf=conf,
                node_dim=len(dataset.generator.node_types),
                edge_dim=len(dataset.generator.edge_types),
                extra_dim=dataset.generator.max_size,
                hidden_activation=self.activation,
                film_dim=None,
            )
            encoder_dim = encoder(batch_fast([sample["graph"]])).shape[-1]
            logger.info(f"graph embedding size: {encoder_dim}")
            return encoder, encoder_dim

        if self.task == "image":
            encoder = ImageEncoder(conf)
            encoder_dim = encoder(sample["image"].unsqueeze(0)).shape[-1]
            logger.info(f"image embedding size: {encoder_dim}")
            return encoder, encoder_dim

        if self.task == "fingerprint":
            encoder = FingerprintEncoder(conf, sample, self.activation)
            encoder_dim = encoder(sample["fingerprint"].unsqueeze(0)).shape[-1]
            logger.info(f"fingerprint embedding size: {encoder_dim}")
            return encoder, encoder_dim

        raise ValueError(f"unknown task: {self.task}")

    @torch.no_grad()
    def param_norm(self):
        params = {
            "param_norm_encoder_target": self.encoder_target.parameters(),
            "param_norm_encoder_query": self.encoder_query.parameters(),
            "param_norm_graph": self.decoder_graph.parameters(),
            "param_norm_edge": self.decoder_edge.parameters(),
        }

        return {
            k: parameters_to_vector(v).norm(2).cpu().item() for k, v in params.items()
        }

    def encode_target(self, target):
        return self.encoder_target(target)

    def encode_query(self, query, conditioning):
        return self.encoder_query(query, conditioning=None)

    def decode(self, targets, subgraphs, terminals):
        readout_graph = torch.cat([targets, subgraphs, terminals], dim=-1)
        readout_edge = torch.cat([targets, subgraphs], dim=-1)

        out_graph = self.decoder_graph(readout_graph)
        out_edge_mask = self.decoder_edge(readout_edge)

        return out_graph, out_edge_mask
