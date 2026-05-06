import io
import itertools
import random
import time
from abc import abstractmethod
from typing import Tuple, Set, ClassVar

import h5py
import matplotlib
import matplotlib.colors
import networkx as nx
import numpy as np
import rustworkx as rx
import torch
from PIL import Image
from loguru import logger
from matplotlib import pyplot
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.patches import Polygon as MplPolygon
from rdkit import Chem
from rdkit.Chem import (
    AtomValenceException,
    rdDepictor,
    rdchem,
)
from rdkit.Chem.Draw import rdMolDraw2D
from scipy.spatial import Delaunay, Voronoi
from shapely.geometry import Polygon, box
from torch.utils.data import Dataset

from src.definitions import Graph
from src.util import (
    is_subgraph_exact,
    get_mol_fingerprint,
    get_graph_fingerprint,
    laplacian_pe,
)


class GraphGenerator:
    supports: ClassVar[Set[str]] = {"graph"}

    def __init__(
        self,
        dataset: str,
        min_size: int,
        max_size: int,
        task: str,
        image_size: int,
        n_node_types: int = None,
        n_edge_types: int = None,
        train: bool = False,
    ):
        self.dataset = dataset
        self.min_size = min_size
        self.max_size = max_size
        self.task = task
        self.image_size = image_size
        self.n_node_types = n_node_types
        self.n_edge_types = n_edge_types
        self.train = train

        error_msg = f"{self.__class__} supports only {self.supports}, got: '{task}'"
        assert task in self.supports, error_msg

        self.train = train

        self.node_types = None
        self.edge_types = None

        self.node_type_idx = None
        self.edge_type_idx = None

        self.node_idx_type = None
        self.edge_idx_type = None

    @abstractmethod
    def get_graph(self, key=None) -> Tuple[nx.Graph, dict]:
        raise NotImplementedError()

    @abstractmethod
    def draw_graph(self, G: nx.Graph, ax):
        raise NotImplementedError()

    @abstractmethod
    def get_frame(self, G: nx.Graph):
        raise NotImplementedError()

    @abstractmethod
    def get_connection(self, target_type, edge_type, node_type):
        raise NotImplementedError()

    @abstractmethod
    def get_root(self, G: nx.Graph):
        raise NotImplementedError()

    def subgraph_match(self, target: rx.PyGraph, query: rx.PyGraph):
        start = time.perf_counter_ns()
        iso = is_subgraph_exact(query, target)
        end = time.perf_counter_ns()
        duration = end - start
        return iso, duration / 1e6

    def filter_nodes(self, G: rx.PyGraph):
        return list(G.node_indices())


class TreeGenerator(GraphGenerator):
    supports: ClassVar[Set[str]] = {"graph", "image", "fingerprint"}

    def __init__(
        self,
        dataset: str,
        min_size: int,
        max_size: int,
        n_node_types: int,
        n_edge_types: int,
        image_size: int,
        task: str,
        train: bool = False,
    ):

        super().__init__(
            dataset,
            min_size,
            max_size,
            task,
            image_size,
            n_node_types,
            n_edge_types,
            train,
        )

        self.markers = ["o", "v", "^", "D", ">", "8", "s", "p", "*", "h", "X"]

        colors = ["black", "red", "green", "blue", "cyan", "magenta", "grey"]

        if n_node_types > len(colors):
            raise ValueError(f"cannot have more than {len(colors)} nodes types")

        if n_edge_types > len(colors):
            raise ValueError(f"cannot have more than {len(colors)} edge types")

        self.node_types = colors[: self.n_node_types]
        self.edge_types = colors[: self.n_edge_types]

        self.range = np.arange(self.min_size, self.max_size + 1)

        self.node_type_idx = {t: i for i, t in enumerate(self.node_types)}
        self.edge_type_idx = {t: i for i, t in enumerate(self.edge_types)}

        self.node_idx_type = {i: t for i, t in enumerate(self.node_types)}
        self.edge_idx_type = {i: t for i, t in enumerate(self.edge_types)}

        self.styles = ["solid"]

        self.fig = pyplot.figure(figsize=(3, 3), dpi=100)
        self.ax = self.fig.add_axes((0, 0, 1, 1))
        self.canvas = FigureCanvasAgg(self.fig)

    def get_connection(self, target_type, edge_type, node_type):
        if target_type is None:
            return {"type": "color", "connection": ("white",) * 3}

        return {"type": "color", "connection": (target_type, edge_type, node_type)}

    def draw_graph(self, G, ax):
        node_types = [n["type"] for _, n in G.nodes(data=True)]
        edge_types = [e["type"] for _, _, e in G.edges(data=True)]
        nx.draw_kamada_kawai(G, ax=ax, node_color=node_types, edge_color=edge_types)

    def get_frame(self, graph):
        self.ax.clear()
        self.ax.set_axis_off()

        layout = nx.kamada_kawai_layout(graph)

        node_colors = [data["type"] for _, data in graph.nodes(data=True)]
        edge_colors = [data["type"] for _, _, data in graph.edges(data=True)]

        nx.draw_networkx_nodes(
            graph, layout, ax=self.ax, node_color=node_colors, node_size=100
        )
        nx.draw_networkx_edges(
            graph, layout, ax=self.ax, edge_color=edge_colors, width=3
        )

        self.canvas.draw()

        rgba = np.frombuffer(self.canvas.buffer_rgba(), dtype=np.uint8)
        rgba = rgba.reshape(self.canvas.get_width_height()[::-1] + (4,))
        rgb_array = rgba[:, :, :3]

        frame = Image.fromarray(rgb_array)
        if frame.size != (self.image_size, self.image_size):
            frame = frame.resize(
                (self.image_size, self.image_size), Image.Resampling.BICUBIC
            )

        return frame

    def get_graph(self, key=None) -> Tuple[nx.Graph, dict]:
        generators = [
            lambda n: nx.random_labeled_tree(n),  # noqa
            # lambda n: nx.barabasi_albert_graph(n, np.random.randint(3, 5)),
        ]

        n_nodes = np.random.choice(self.range)
        graph = np.random.choice(generators)(n_nodes)

        for node in graph.nodes:
            graph.nodes[node]["type"] = random.choice(self.node_types)

        for edge in graph.edges:
            graph.edges[edge]["type"] = random.choice(self.edge_types)

        info = {}
        if self.task == "image":
            info["image"] = np.array(self.get_frame(graph))

        if self.task == "fingerprint":
            info["fingerprint"] = get_graph_fingerprint(graph)

        return graph, info

    def get_root(self, G):
        # can be any node
        return 0


class ColoringGenerator(GraphGenerator):
    supports: ClassVar[Set[str]] = {"graph", "image", "fingerprint"}

    def __init__(
        self,
        dataset: str,
        min_size: int,
        max_size: int,
        n_node_types: int,
        n_edge_types: int,
        image_size: int,
        task: str,
        train: bool = False,
    ):
        super().__init__(
            dataset,
            min_size,
            max_size,
            task,
            image_size,
            n_node_types,
            n_edge_types,
            train,
        )

        if self.n_node_types < 4:
            raise ValueError("n_node_types must be >= 4 (Four Color Theorem).")

        if self.n_edge_types != 1:
            raise ValueError("n_edge_types must be 1.")

        self.cmap = pyplot.get_cmap("tab10")
        self.bounding_box = box(0, 0, 1, 1)

        self.node_types = [
            matplotlib.colors.to_hex(self.cmap(i)) for i in range(self.n_node_types)
        ]
        self.edge_types = ["#000000"]

        self.node_type_idx = {t: i for i, t in enumerate(self.node_types)}
        self.edge_type_idx = {t: i for i, t in enumerate(self.edge_types)}

        self.fig = pyplot.figure(figsize=(3, 3), dpi=100)
        self.ax = self.fig.add_axes((0, 0, 1, 1))
        self.canvas = FigureCanvasAgg(self.fig)

    def get_graph(self, key=None) -> Tuple[nx.Graph, dict]:
        while True:
            n = random.randint(self.min_size, self.max_size)
            points = np.random.rand(n, 2)
            tri = Delaunay(points)
            G = nx.Graph()

            for i, p in enumerate(points):
                G.add_node(i, pos=p)

            for s in tri.simplices:
                G.add_edges_from([(s[0], s[1]), (s[1], s[2]), (s[2], s[0])])

            color_map = nx.coloring.greedy_color(G, strategy="largest_first")

            if not color_map or max(color_map.values()) < self.n_node_types:
                # randomly permute colors
                available_colors = list(range(self.n_node_types))
                random.shuffle(available_colors)

                # if more colors are available than used, split some color classes
                used_colors = max(color_map.values()) + 1
                if used_colors < self.n_node_types:
                    # we want to use as many as possible
                    target_n_colors = random.randint(used_colors, self.n_node_types)

                    # color_map has values 0..used_colors-1
                    # we want to re-assign some nodes to colors used_colors..target_n_colors-1
                    # to maintain a valid coloring, we can only split existing color classes
                    nodes_by_color = {}
                    for node, c in color_map.items():
                        nodes_by_color.setdefault(c, []).append(node)

                    current_max_color = used_colors - 1
                    while current_max_color < target_n_colors - 1:
                        # pick a color class to split
                        # prefer larger ones
                        c_to_split = max(
                            nodes_by_color.keys(), key=lambda k: len(nodes_by_color[k])
                        )
                        if len(nodes_by_color[c_to_split]) < 2:
                            break  # cannot split anymore

                        # split it
                        nodes = nodes_by_color[c_to_split]
                        random.shuffle(nodes)
                        split_point = len(nodes) // 2

                        new_color = current_max_color + 1
                        nodes_by_color[new_color] = nodes[split_point:]
                        nodes_by_color[c_to_split] = nodes[:split_point]

                        for node in nodes_by_color[new_color]:
                            color_map[node] = new_color

                        current_max_color = new_color

                for node, c_idx in color_map.items():
                    G.nodes[node]["type"] = self.node_types[available_colors[c_idx]]

                for u, v in G.edges():
                    G.edges[u, v]["type"] = self.edge_types[0]

                info = {}
                if self.task == "image":
                    info["image"] = np.array(self.get_frame(G))

                if self.task == "fingerprint":
                    info["fingerprint"] = get_graph_fingerprint(G)

                return G, info

    def get_frame(self, G) -> Image.Image:
        self.ax.clear()
        self.ax.set_axis_off()

        if (
            self.task == "image"
            and len(G) > 1
            and all("pos" in G.nodes[n] for n in G.nodes())
        ):
            node_list = list(G.nodes())
            points = np.array([G.nodes[n]["pos"] for n in node_list])

            dummy = np.array([[-10, -10], [10, -10], [10, 10], [-10, 10]])
            vor = Voronoi(np.vstack([points, dummy]))

            self.ax.set_xlim(0, 1)
            self.ax.set_ylim(0, 1)

            for i in range(len(points)):
                region_idx = vor.point_region[i]
                vertices = vor.vertices[vor.regions[region_idx]]
                if -1 in vor.regions[region_idx]:
                    continue

                poly = Polygon(vertices).intersection(self.bounding_box)
                if not poly.is_empty:
                    color = G.nodes[node_list[i]]["type"]
                    geoms = [poly] if poly.geom_type == "Polygon" else poly.geoms
                    for p in geoms:
                        self.ax.add_patch(
                            MplPolygon(
                                np.array(p.exterior.coords),
                                facecolor=color,
                                edgecolor="black",
                                linewidth=1.0,
                            )
                        )
        else:
            self.draw_graph(G, self.ax)

        self.canvas.draw()
        w, h = self.fig.canvas.get_width_height()
        buf = np.frombuffer(self.canvas.buffer_rgba(), dtype=np.uint8).reshape(h, w, 4)

        img = Image.fromarray(buf[:, :, :3])
        return img.resize((self.image_size, self.image_size), Image.Resampling.BICUBIC)

    def draw_graph(self, G, ax):
        node_colors = [G.nodes[n]["type"] for n in G.nodes()]
        nx.draw_kamada_kawai(
            G, ax=ax, node_color=node_colors, node_size=100, edge_color="black"
        )

    def get_connection(self, target_type, edge_type, node_type):
        if target_type is None:
            return {"type": "color", "connection": ("white",) * 3}
        return {"type": "color", "connection": (target_type, edge_type, node_type)}

    def get_root(self, G):
        return 0


class MoleculeGenerator(GraphGenerator):
    supports: ClassVar[Set[str]] = {"graph", "image", "fingerprint"}

    def __init__(
        self,
        dataset: str,
        min_size: int,
        max_size: int,
        image_size: int,
        train: bool,
        task: str,
    ):

        dataset = dataset.split("_")[0]
        super().__init__(dataset, min_size, max_size, task, image_size, train)
        self.h5_file = f"data/{self.dataset}_data.h5"
        self.h5_buckets = [str(i) for i in range(self.min_size, self.max_size + 1)]
        self.h5_buckets_sizes = []
        self.h5_buckets_probs = []

        self.h5_data = None

        self.train_idx_ranges = {}
        self.val_idx_ranges = {}

        self.node_label_idx = torch.load(
            f"data/{self.dataset}_mapping.pt", weights_only=False
        )
        self.node_type_idx = dict(
            zip(self.node_label_idx.keys(), range(len(self.node_label_idx.keys())))
        )

        self.max_degree = 4

        logger.info(f"using types: {list(self.node_label_idx.values())}")

        self.edge_type_idx = {
            1: 0,  # single
            2: 1,  # double
            3: 2,  # triple
        }

        self.edge_label_idx = {
            1: "-",  # single
            2: "=",  # double
            3: r"$\equiv$",  # triple
        }

        self.node_types = list(self.node_type_idx.keys())
        self.edge_types = list(self.edge_type_idx.keys())

    def filter_nodes(self, G: rx.PyGraph):
        return [i for i in G.node_indices() if G.degree(i) <= self.max_degree]

    def get_connection(self, u_type, e_type, v_type):
        if u_type is None:
            return {"type": "text", "connection": ("", "", "")}

        u = self.node_label_idx[u_type]
        e = self.edge_label_idx[e_type]
        v = self.node_label_idx[v_type]

        return {"type": "text", "connection": (u, e, v)}

    def smiles_to_nx(self, smiles):
        mol = Chem.MolFromSmiles(smiles, sanitize=True)

        Chem.Kekulize(mol, clearAromaticFlags=True)

        G = nx.Graph()

        for atom in mol.GetAtoms():
            G.add_node(
                atom.GetIdx(), type=atom.GetAtomicNum(), chiral=atom.GetChiralTag()
            )

        bond_map = {
            rdchem.BondType.SINGLE: 1,
            rdchem.BondType.DOUBLE: 2,
            rdchem.BondType.TRIPLE: 3,
        }

        bond_dir_map = {
            rdchem.BondDir.NONE: 0,
            rdchem.BondDir.BEGINWEDGE: 1,
            rdchem.BondDir.BEGINDASH: 2,
            rdchem.BondDir.ENDUPRIGHT: 3,
            rdchem.BondDir.ENDDOWNRIGHT: 4,
            rdchem.BondDir.EITHERDOUBLE: 5,
        }

        for bond in mol.GetBonds():
            G.add_edge(
                bond.GetBeginAtomIdx(),
                bond.GetEndAtomIdx(),
                type=bond_map[bond.GetBondType()],
                stereo=bond_dir_map[bond.GetBondDir()],
                begin_atom=bond.GetBeginAtomIdx(),
            )

        return G

    def nx_to_rdkit(self, graph: nx.Graph):
        mol = Chem.RWMol()
        node_to_idx = {}

        for node, data in graph.nodes(data=True):
            atom = Chem.Atom(data["type"])
            if "chiral" in data:
                atom.SetChiralTag(data["chiral"])
            idx = mol.AddAtom(atom)
            node_to_idx[node] = idx

        bond_map = {
            1: Chem.BondType.SINGLE,
            2: Chem.BondType.DOUBLE,
            3: Chem.BondType.TRIPLE,
        }

        dir_map = {
            0: Chem.BondDir.NONE,
            1: Chem.BondDir.BEGINWEDGE,
            2: Chem.BondDir.BEGINDASH,
            3: Chem.BondDir.ENDUPRIGHT,
            4: Chem.BondDir.ENDDOWNRIGHT,
            5: Chem.BondDir.EITHERDOUBLE,
        }

        for u, v, data in graph.edges(data=True):
            u_idx, v_idx = node_to_idx[u], node_to_idx[v]

            start_node = data["begin_atom"] if "begin_atom" in data else u
            if start_node == v:
                u_idx, v_idx = v_idx, u_idx

            b_type = data["type"] if "type" in data else 1
            mol.AddBond(u_idx, v_idx, bond_map[b_type])

            stereo_val = data["stereo"] if "stereo" in data else 0
            new_bond = mol.GetBondBetweenAtoms(u_idx, v_idx)
            new_bond.SetBondDir(dir_map[stereo_val])

        mol = mol.GetMol()

        for atom in mol.GetAtoms():
            atom.SetFormalCharge(0)

        return mol

    def get_graph(self, key=None) -> Tuple[nx.Graph, dict]:
        if self.h5_data is None:
            self.h5_data = h5py.File(self.h5_file, "r")
            for bucket in self.h5_buckets:
                bucket_size = len(self.h5_data[bucket]["data"])
                split_idx = int(0.9 * bucket_size)

                self.train_idx_ranges[bucket] = (0, split_idx)
                self.val_idx_ranges[bucket] = (split_idx, bucket_size)
                self.h5_buckets_sizes.append(bucket_size)

            self.h5_buckets_sizes = np.array(self.h5_buckets_sizes)
            self.h5_buckets_probs = self.h5_buckets_sizes / self.h5_buckets_sizes.sum()

        idx = self.train_idx_ranges if self.train else self.val_idx_ranges

        if key is None:
            bucket = np.random.choice(self.h5_buckets, p=self.h5_buckets_probs)
            bucket_range = idx[bucket]
            bucket_idx = np.random.randint(*bucket_range)
        else:
            bucket, bucket_idx = key

        smiles = self.h5_data[bucket]["data"][bucket_idx].decode("utf8")
        graph = self.smiles_to_nx(smiles)

        info = {"id": (int(bucket), bucket_idx)}

        if self.task == "image":
            info["image"] = np.array(self.get_frame(graph))

        if self.task == "fingerprint":
            info["fingerprint"] = get_mol_fingerprint(smiles)

        return graph, info

    def get_drawer(self):
        drawer = rdMolDraw2D.MolDraw2DCairo(self.image_size, self.image_size)
        options = drawer.drawOptions()

        options.useBWAtomPalette()

        options.addAtomIndices = False
        options.addStereoAnnotation = False
        options.includeAtomTags = False
        options.clearBackground = False
        options.continuousHighlight = True
        options.explicitMethyl = False

        if self.train:
            options.fixedBondLength = int(np.random.randint(30, 50))
            options.bondLineWidth = float(np.random.uniform(1.0, 2.0))
            options.padding = float(np.random.uniform(0.1, 0.2))
            options.baseFontSize = float(np.random.uniform(0.5, 0.9))
            options.minFontSize = int(np.random.randint(8, 12))

            if np.random.rand() < 0.5:
                options.comicMode = True

            options.rotate = float(np.random.uniform(0, 360))

        return drawer

    def draw_mol_frame(self, graph: nx.Graph):
        mol = self.nx_to_rdkit(graph)

        Chem.AssignStereochemistry(mol, force=True, cleanIt=True)

        rdDepictor.Compute2DCoords(mol, canonOrient=False)
        rdDepictor.StraightenDepiction(mol)

        Chem.WedgeMolBonds(mol, mol.GetConformer())
        rdMolDraw2D.PrepareMolForDrawing(mol, kekulize=True, addChiralHs=False)

        drawer = self.get_drawer()
        drawer.DrawMolecule(mol)
        drawer.FinishDrawing()

        image = Image.open(io.BytesIO(drawer.GetDrawingText()))

        rgb = Image.new("RGB", image.size, (255, 255, 255))
        rgb.paste(image, mask=image.getchannel("A"))
        return rgb

    def draw_graph(self, G, ax):
        try:
            img = self.draw_mol_frame(G)
            ax.imshow(img)
        except AtomValenceException:
            ax.text(x=0.5, y=0.5, s="invalid mol", fontsize=24)

    def get_frame(self, graph):
        return self.draw_mol_frame(graph)

    def get_root(self, G):
        return 0


class GraphDataset(Dataset):
    def __init__(self, generator: GraphGenerator):
        self.generator = generator

        mask = []

        for u, e, v in itertools.product(
            self.generator.node_types,
            self.generator.edge_types,
            self.generator.node_types,
        ):
            mask.append(self.make_edge_mask(u, e, v))

        self.mask = sorted(list(set(mask)))

        self.mask_to_id = {mask: i for i, mask in enumerate(self.mask)}

    def make_edge_mask(self, u_type=None, e_type=None, v_type=None):
        # make pair undirected, directed makes handling less clear
        u_type, v_type = sorted((u_type, v_type))
        return u_type, e_type, v_type

    def make_node_mask(self, u=None, u_type=None, e_type=None, v=None, v_type=None):
        # types matter as a particular node id can be added via multiple types
        return u, u_type, e_type, v, v_type

    def get_edge_filter_from_mask(self, mask):
        idx = set(torch.where(mask > 0.1)[0].tolist())
        return set([self.mask[i] for i in idx])

    def rx_to_nx(self, rx_graph: rx.PyGraph) -> nx.Graph:
        nx_graph = nx.Graph()
        for idx, data in enumerate(rx_graph.nodes()):
            nx_graph.add_node(idx, **data)
        for u, v in rx_graph.edge_list():
            edge_data = rx_graph.get_edge_data(u, v)
            nx_graph.add_edge(u, v, **edge_data)

        nx_graph.graph = rx_graph.attrs
        return nx_graph

    def rx_to_pt(self, graph: rx.PyGraph):
        nodes = torch.tensor(
            [self.generator.node_type_idx[node["type"]] for node in graph.nodes()],
            dtype=torch.int64,
        )

        degrees = torch.tensor(
            [graph.degree(i) for i in graph.node_indices()], dtype=torch.int32
        )

        if graph.num_edges() > 0:
            edges = torch.tensor(graph.edge_list()).T
            edges = torch.cat([edges, edges.flip(0)], dim=1)
        else:
            edges = torch.empty((2, 0), dtype=torch.int64)

        edges_f = torch.tensor(
            [
                self.generator.edge_type_idx[graph.get_edge_data(u, v)["type"]]
                for u, v in graph.edge_list()
            ],
            dtype=torch.int64,
        )
        edges_f = edges_f.unsqueeze(0).expand(2, -1).reshape(-1)

        terminal = torch.tensor(graph.attrs["terminal"])

        extras = laplacian_pe(graph, self.generator.max_size)

        assert edges.shape[1] == edges_f.shape[0], "mismatch between number of edges"

        return Graph(
            nodes=nodes,
            degrees=degrees,
            edges=edges,
            edges_f=edges_f,
            terminal=terminal,
            extra=extras,
        )

    def copy(self, G):
        H = nx.Graph()
        H.add_nodes_from(G.nodes(data=True))
        H.add_edges_from(G.edges(data=True))
        H.graph.update(G.graph)  # noqa
        return H

    def bfs_neighbours_by_depth(self, graph: rx.PyGraph, depth: int):
        dist_map = rx.all_pairs_dijkstra_path_lengths(graph, lambda _: 1.0)
        return {
            u: {v for v, d in dist.items() if d <= depth}
            for u, dist in dist_map.items()
        }

    def fireforest_sample(self, graph: rx.PyGraph, n: int, p: Tuple[float, float]):
        p = np.random.uniform(*p)
        start = np.random.randint(graph.num_nodes())
        visited = {start}
        frontier = [start]

        while frontier and len(visited) < n:
            u = frontier.pop(random.randrange(len(frontier)))
            nbrs = list(graph.neighbors(u))
            random.shuffle(nbrs)

            for v in nbrs:
                if len(visited) >= n:
                    break
                if v in visited:
                    continue
                if random.random() < p:
                    visited.add(v)
                    frontier.append(v)

            if not frontier and len(visited) < n:
                boundary = set()
                for u in visited:
                    boundary.update(graph.neighbors(u))
                boundary = [v for v in boundary if v not in visited]
                if not boundary:
                    break
                v = random.choice(boundary)
                visited.add(v)
                frontier.append(v)

        return graph.subgraph(list(visited))

    def successors_rx(self, G: rx.PyGraph, edges_mask, nodes_mask, terminal=True):
        successors = []

        if G.num_nodes() == 0:
            for node_type in self.generator.node_types:
                G_new = G.copy()
                G_new.add_node({"type": node_type})
                G_new.attrs = {
                    "terminal": False,
                    "connection": self.generator.get_connection(None, None, None),
                    "action": None,
                }
                successors.append(G_new)

            return successors

        filtered_nodes = self.generator.filter_nodes(G)

        missing_nodes = []

        for u in filtered_nodes:
            u_type = G[u]["type"]

            for v_type, e_type in itertools.product(
                self.generator.node_types, self.generator.edge_types
            ):
                missing_nodes.append((u, u_type, e_type, v_type))

        for u, u_type, e_type, v_type in missing_nodes:
            edge_mask = self.make_edge_mask(u_type=u_type, e_type=e_type, v_type=v_type)

            if edge_mask not in edges_mask:
                continue

            G_new = G.copy()
            v = G_new.add_node({"type": v_type})
            G_new.add_edge(u, v, {"type": e_type})

            G_new.attrs = {
                "terminal": False,
                "connection": self.generator.get_connection(u_type, e_type, v_type),
                "action": None,
            }

            successors.append(G_new)

        existing_nodes = set(filtered_nodes)
        existing_edges = set(G.edge_list())
        all_possible_edges = set(itertools.combinations(existing_nodes, 2))
        missing_edges = list(all_possible_edges - existing_edges)

        missing_edges = list(
            itertools.product(missing_edges, self.generator.edge_types)
        )

        for (u, v), e_type in missing_edges:
            u_type = G[u]["type"]
            v_type = G[v]["type"]

            edge_mask = self.make_edge_mask(u_type=u_type, e_type=e_type, v_type=v_type)
            node_mask = self.make_node_mask(
                u=u, u_type=u_type, e_type=e_type, v=v, v_type=v_type
            )

            if edge_mask not in edges_mask:
                continue

            if node_mask in nodes_mask:
                continue

            G_new = G.copy()
            G_new.add_edge(u, v, {"type": e_type})

            G_new.attrs = {
                "terminal": False,
                "connection": self.generator.get_connection(u_type, e_type, v_type),
                "action": node_mask,
            }

            successors.append(G_new)

        if terminal:
            G_term = G.copy()
            G_term.attrs = {
                "terminal": True,
                "connection": self.generator.get_connection(None, None, None),
                "action": None,
            }

            successors.append(G_term)

        return successors

    def get_item(self, key=None, info=False) -> rx.PyGraph:
        graph_nx, graph_info = None, None

        while True:
            try:
                graph_nx, graph_info = self.generator.get_graph(key)
                break
            except Exception as e:
                logger.error(f"failed to get graph: {e}")
                continue

        graph_rx = rx.networkx_converter(graph_nx, keep_attributes=True)

        graph_rx.attrs = {
            "terminal": True,
            "edge_counts": self.get_edge_counts(graph_rx),
        }

        if "image" in graph_info:
            frame = torch.from_numpy(graph_info["image"])
            graph_rx.attrs["image"] = frame.permute(2, 0, 1) / 255.0

        if "fingerprint" in graph_info:
            graph_rx.attrs["fingerprint"] = torch.from_numpy(graph_info["fingerprint"])

        return graph_rx if not info else (graph_rx, graph_info)

    def get_edge_counts(self, graph: rx.PyGraph):
        counts = torch.zeros(len(self.mask), dtype=torch.int32)

        for u, v in graph.edge_list():
            mask = self.make_edge_mask(
                graph[u]["type"],
                graph.get_edge_data(u, v)["type"],
                graph[v]["type"],
            )
            mask_id = self.mask_to_id[mask]
            counts[mask_id] += 1

        return counts

    def get_edge_mask(self, target: rx.PyGraph, subgraph: rx.PyGraph):
        target_counts = target.attrs["edge_counts"]

        if subgraph.num_nodes() == 1:
            return (target_counts > 0).float()

        sub_counts = self.get_edge_counts(subgraph)

        return (target_counts > sub_counts).float()

    def is_same_size(self, query: rx.PyGraph, target: rx.PyGraph):
        nodes_match = query.num_nodes() == target.num_nodes()
        edges_match = query.num_edges() == target.num_edges()
        return nodes_match and edges_match

    def get_single_sample(self, target: rx.PyGraph, query: rx.PyGraph):
        successor_valid, successor_time = self.generator.subgraph_match(target, query)

        if query.attrs["terminal"]:
            successor_valid = successor_valid and self.is_same_size(query, target)

        query.attrs["valid"] = successor_valid
        query.attrs["time"] = successor_time

        subgraph_edge_mask = self.get_edge_mask(target, query)

        query_pt = self.rx_to_pt(query)

        sample = {
            "graph": query_pt,
            "terminal": torch.tensor([query.attrs["terminal"]], dtype=torch.float32),
            "label": torch.tensor([successor_valid], dtype=torch.float32),
            "edge_mask": subgraph_edge_mask,
            "time": torch.tensor([successor_time], dtype=torch.float32),
        }

        if "image" in target.attrs:
            sample["image"] = target.attrs["image"]

        if "fingerprint" in target.attrs:
            sample["fingerprint"] = target.attrs["fingerprint"]

        return sample

    def get_root(self, graph: nx.Graph):
        return self.generator.get_root(graph)
