import itertools

import networkx as nx
import rustworkx as rx
import torch
from PIL import Image, ImageChops
from hydra import initialize, compose
from hydra.utils import instantiate
from matplotlib import pyplot as plt
from tqdm import tqdm

from src.data import GraphDataset
from src.util import batch_fast


def crop_frame(frame: Image.Image):
    # find bounding box of non-white pixels
    bg = Image.new(frame.mode, frame.size, (255, 255, 255))
    diff = ImageChops.difference(frame, bg)
    bbox = diff.getbbox()

    if not bbox:
        return frame

    # add some padding
    padding = 10
    left, upper, right, lower = bbox
    left = max(0, left - padding)
    upper = max(0, upper - padding)
    right = min(frame.size[0], right + padding)
    lower = min(frame.size[1], lower + padding)

    return frame.crop((left, upper, right, lower))


def is_same_graph(target, pred):
    return nx.is_isomorphic(
        target,
        pred,
        node_match=lambda n1, n2: n1["type"] == n2["type"],
        edge_match=lambda e1, e2: e1["type"] == e2["type"],
    )


def plot_graphs(target, pred, graphs, dataset):
    if is_same_graph(target, pred):
        pred = target

    cols = len(graphs) + 2  # target + all decoded variants

    fig, axes = plt.subplots(1, cols, squeeze=False, figsize=(3 * cols, 4))
    fig.subplots_adjust(left=0, right=1, top=1, bottom=0, wspace=0.02, hspace=0.02)

    target_frame = dataset.generator.get_frame(target)
    axes[0, 0].imshow(target_frame)
    axes[0, 0].set_axis_off()

    pred_frame = dataset.generator.get_frame(pred)
    axes[0, 1].imshow(pred_frame)
    axes[0, 1].set_axis_off()

    for i, g in enumerate(graphs, start=2):
        if is_same_graph(target, g):
            g = target

        frame = dataset.generator.get_frame(g)
        axes[0, i].imshow(frame)
        axes[0, i].set_axis_off()

    plt.tight_layout()
    plt.show()


@torch.inference_mode()
def decode_embedding(dataset, model, embedding):
    successors_rx = []

    for t in dataset.generator.node_types:
        G = rx.PyGraph()
        G.add_node({"type": t})
        G.attrs = {"terminal": False}
        successors_rx.append(G)

    for _ in itertools.count():
        queries_batch = batch_fast([dataset.rx_to_pt(g) for g in successors_rx])
        queries_batch = {k: v.to(device) for k, v in queries_batch.items()}

        targets_embeddings = embedding.expand(len(successors_rx), -1)
        queries_embeddings = model.encode_query(queries_batch, targets_embeddings)
        queries_terminals = queries_batch["terminals"]

        preds_graph, preds_edge = model.decode(
            targets_embeddings, queries_embeddings, queries_terminals
        )
        idx = preds_graph.reshape(-1).argmax().item()

        graph_prediction = successors_rx[idx]
        # TODO add action filter, too

        if graph_prediction.attrs["terminal"]:
            return graph_prediction

        edge_prediction = preds_edge[idx]
        edge_mask = dataset.get_edge_filter_from_mask(edge_prediction)

        successors_rx = dataset.successors_rx(
            graph_prediction, edges_mask=edge_mask, nodes_mask=set(), terminal=True
        )


# the latent space of trees is very well-structured in terms of colors
# for coloring, the learned representation is not as composable!
dataset_name = "pubchem_32"
task_name = "graph"

with initialize(version_base=None, config_path="conf"):
    conf = compose(config_name="prod_hk", overrides=[f"environment={dataset_name}"])

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model_path = f"models/{dataset_name}_{task_name}.pt"
model = torch.load(model_path, weights_only=False).to(device)
model.eval()

conf.environment.task = task_name
generator = instantiate(conf.environment, train=False)
dataset = GraphDataset(generator)

target_graph_rx = dataset.get_item()


inter_graph_rx = dataset.get_item()

target_graph_rx.attrs = {"terminal": False}
inter_graph_rx.attrs = {"terminal": False}

target_graph_pt = dataset.rx_to_pt(target_graph_rx)
inter_graph_pt = dataset.rx_to_pt(inter_graph_rx)

batch = batch_fast([target_graph_pt, inter_graph_pt])
batch = {k: v.to(device) for k, v in batch.items()}

target_emb, inter_graph_emb = model.encode_target(batch)

for target_scale in [3.0]:
    n_interpolations = 8
    upper_emb = target_emb * target_scale
    lower_emb = inter_graph_emb * 0

    t = torch.linspace(
        0.0, 1.0, n_interpolations, device=target_emb.device, dtype=target_emb.dtype
    )

    # TODO try using PCA to find the important dimensions
    x = 0
    y = 512

    if not (0 <= x < y <= target_emb.shape[0]):
        raise ValueError(
            f"Invalid interpolation range: x={x}, y={y}, dim={target_emb.shape[0]}"
        )

    interpolations = lower_emb[None, :].expand(n_interpolations, -1).clone()
    interpolations[:, x:y] = (1 - t[:, None]) * lower_emb[None, x:y] + t[
        :, None
    ] * upper_emb[None, x:y]

    split_idx = n_interpolations // 2
    interpolations = torch.cat(
        [
            target_emb[None, :],
            interpolations[:split_idx],
            target_emb[None, :],
            interpolations[split_idx:],
        ],
        dim=0,
    )

    decoded = [decode_embedding(dataset, model, emb) for emb in tqdm(interpolations)]
    decoded = [dataset.rx_to_nx(g) for g in decoded]

    plot_graphs(dataset.rx_to_nx(target_graph_rx), decoded[0], decoded[1:], dataset)
