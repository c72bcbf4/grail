import sys

import igraph as ig
import pandas as pd
import torch
from hydra import initialize, compose
from torch.nn.functional import binary_cross_entropy_with_logits
from torchmetrics import MeanMetric
from tqdm import trange

from src.data import GraphDataset, TreeGenerator, ColoringGenerator, MoleculeGenerator
from src.nn import MultiLayerPerceptron, GraphEncoder
from src.util import batch_fast


def get_datasets(name, size):
    if name == "coloring":
        gen = ColoringGenerator(
            dataset="trees",
            min_size=size,
            max_size=size,
            train=True,
            task="graph",
            image_size=128,
            n_node_types=4,
            n_edge_types=1,
        )
        return gen, gen

    if name == "trees":
        gen = TreeGenerator(
            dataset="trees",
            min_size=size,
            max_size=size,
            train=True,
            task="graph",
            image_size=128,
            n_node_types=1,
            n_edge_types=1,
        )
        return gen, gen

    if name == "qm9":
        generator_args = dict(
            dataset="qm9",
            min_size=size,
            max_size=size,
            image_size=128,
            task="graph",
        )
        train = MoleculeGenerator(**generator_args, train=True)
        test = MoleculeGenerator(**generator_args, train=False)
        return train, test

    raise ValueError(f"invalid dataset {name}")


def bfs_relabel(g):
    degrees = g.degree()
    root = max(range(g.vcount()), key=lambda i: degrees[i])

    visited = [False] * g.vcount()
    order = []
    queue = [root]
    visited[root] = True

    while queue:
        v = queue.pop(0)
        order.append(v)

        # get neighbors that are not visited
        neighbors = [n for n in g.neighbors(v) if not visited[n]]
        # sort neighbors by degree descending
        neighbors.sort(key=lambda x: g.degree(x), reverse=True)

        for n in neighbors:
            visited[n] = True
        queue.extend(neighbors)

    # create mapping old -> new
    mapping = {old: new for new, old in enumerate(order)}

    # relabel graph
    g_relabel = g.copy()
    g_relabel.vs["name"] = [mapping[v.index] for v in g.vs]
    g_relabel = g_relabel.permute_vertices(list(mapping.values()))

    return g_relabel


def get_relabeled_graph(graph, labeling):
    if labeling == "random":
        return graph

    if labeling == "canonical":
        perm = graph.canonical_permutation()
        graph = graph.permute_vertices(perm)
        return graph
    if labeling == "bfs":
        return bfs_relabel(graph)

    raise ValueError(f"invalid labeling '{labeling}'")


def get_batch(dataset, batch_size, device, labeling):
    x, y = [], []
    for _ in range(batch_size):
        graph_rx = dataset.get_item()
        graph_nx = dataset.rx_to_nx(graph_rx)

        graph_ig = ig.Graph.from_networkx(graph_nx)
        graph_ig = get_relabeled_graph(graph_ig, labeling)

        adj_mat = torch.tensor(graph_ig.get_adjacency().data)

        x.append(dataset.rx_to_pt(graph_rx))
        y.append(adj_mat)

    x = batch_fast(x, device=device)
    y = torch.stack(y).to(device)

    return x, y.float()


def run_experiment(dataset, labeling, size):
    with initialize(version_base=None, config_path="conf"):
        conf = compose(config_name="dev")

    generator_train, generator_test = get_datasets(dataset, size=size)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dataset_train = GraphDataset(generator=generator_train)
    dataset_test = GraphDataset(generator=generator_test)

    activation = torch.nn.ReLU()

    encoder = GraphEncoder(
        conf.encoder_target,
        node_dim=len(dataset_train.generator.node_types),
        edge_dim=len(dataset_train.generator.edge_types),
        hidden_activation=activation,
        film_dim=None,
    )

    decoder = MultiLayerPerceptron(
        input_size=conf.encoder_target.hidden_channels,
        hidden_channels=[512, 512],
        output_size=size**2,
        norm="LayerNorm",
        hidden_activation=activation,
    )

    encoder = encoder.to(device)
    decoder = decoder.to(device)

    optimizer = torch.optim.Adam(decoder.parameters(), lr=1e-3)

    epochs = 10
    samples_per_epoch = 100_000
    batch_size = 32
    steps_per_epoch = samples_per_epoch // batch_size

    for epoch in range(epochs):
        metric_loss = MeanMetric().to(device)

        for _ in trange(
            steps_per_epoch, file=sys.stdout, leave=False, desc=f"epoch {epoch}"
        ):
            x, y = get_batch(
                dataset_train, batch_size=batch_size, device=device, labeling=labeling
            )

            encoded = encoder(x)
            decoded = decoder(encoded).reshape(-1, size, size)

            loss = binary_cross_entropy_with_logits(decoded, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            metric_loss.update(loss)

        x, y = get_batch(
            dataset_test, batch_size=1_000, device=device, labeling=labeling
        )
        encoded = encoder(x)
        decoded = decoder(encoded)

        y = y.flatten(1) > 0.5
        decoded = decoded.flatten(1) > 0.5
        correct = (y == decoded).all(-1).float().mean()

        print(
            f"{dataset}/{labeling}: epoch={epoch}: loss={metric_loss.compute().item():.4f}, correct={correct:.4f}"
        )

    with torch.no_grad():
        x, y = get_batch(
            dataset_test, batch_size=10_000, device=device, labeling=labeling
        )
        encoded = encoder(x)
        decoded = decoder(encoded)

        loss = binary_cross_entropy_with_logits(decoded.flatten(1), y.flatten(1))

        y = y.flatten(1) > 0.5
        decoded = decoded.flatten(1) > 0.5
        correct = (y == decoded).all(-1).float().mean()

        return round(loss.item(), 4), round(correct.item(), 4)


all_results = []

experiments = [
    # ("coloring", "random"),
    # ("coloring", "bfs"),
    # ("coloring", "canonical"),
    # ("trees", "random"),
    # ("trees", "bfs"),
    # ("trees", "canonical"),
    # ("qm9", "random"),
    # ("qm9", "bfs"),
    ("qm9", "canonical"),
]


for dataset, labeling in experiments:
    loss, correct = run_experiment(size=9, dataset=dataset, labeling=labeling)

    log = dict(dataset=dataset, labelling=labeling, loss=loss, correct=correct)
    all_results.append(log)

    out = pd.DataFrame(all_results).to_string(index=False)
    print(out)
