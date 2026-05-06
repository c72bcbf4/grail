import os
import random
from collections import defaultdict
from multiprocessing import Pool

import h5py
import networkx as nx
import torch
from rdkit import Chem, RDLogger
from tqdm import tqdm

RDLogger.DisableLog("rdApp.*")

# fmt: off
valid = {5,6,7,8,9,11,12,13,14,15,16,17,19,20,24,25,26,27,28,29,30,33,34,35,50,52,53,56,78,80,82}
# fmt: on


def smiles_to_nx(smiles):
    mol = Chem.MolFromSmiles(smiles, sanitize=True)

    if mol is None:
        return None

    mol_types = {atom.GetAtomicNum() for atom in mol.GetAtoms()}

    diff = mol_types - valid

    if len(diff) > 0:
        return None

    G = nx.Graph()
    for atom in mol.GetAtoms():
        G.add_node(atom.GetIdx(), type=atom.GetAtomicNum())

    for bond in mol.GetBonds():
        G.add_edge(
            bond.GetBeginAtomIdx(),
            bond.GetEndAtomIdx(),
            type=int(bond.GetBondTypeAsDouble()),
        )

    if not nx.is_connected(G):
        return None

    return G


index_map = f"data/pubchem_mapping.pt"

table = Chem.GetPeriodicTable()
mapping = {t: table.GetElementSymbol(t) for t in sorted(valid)}

torch.save(mapping, index_map)

source = "data/pubchem_data"
target = "data/pubchem_data.h5"
source_files = [os.path.join(source, f) for f in os.listdir(source) if "part_" in f]
random.shuffle(source_files)

print(f"processing {len(source_files)} files")


def process_file(filename):
    batch_types = set()
    batch = defaultdict(list)

    n_valid = 0
    n_done = 0

    with open(filename, "r") as file:
        for line in file:
            smiles = line.strip().split("\t")[-1]

            G = smiles_to_nx(smiles)

            n_done += 1

            if G is None:
                continue

            n_valid += 1

            graph_types = [data["type"] for _, data in G.nodes(data=True)]

            batch_types.update(graph_types)

            batch[len(G)].append(smiles)

    return batch, all_types, n_valid, n_done


n_parallel = 48

total = len(source_files)
all_types = set()

total_valid = 0
total_done = 0

with (
    Pool(n_parallel) as pool,
    h5py.File(target, "a") as file,
    tqdm(total=total) as pbar,
):
    for batch, types, n_valid, n_done in pool.imap_unordered(
        process_file, source_files, chunksize=1
    ):
        all_types.update(types)

        for size, smiles_list in batch.items():
            group_name = str(size)
            grp = file.require_group(group_name)

            if grp.get("data") is None:
                dset = grp.create_dataset(
                    "data",
                    shape=(0,),
                    maxshape=(None,),
                    dtype=h5py.string_dtype(encoding="utf-8"),
                    chunks=True,
                )
            else:
                dset = grp["data"]

            old_size = dset.shape[0]
            new_size = old_size + len(smiles_list)
            dset.resize((new_size,))
            dset[old_size:] = smiles_list

        total_valid += n_valid
        total_done += n_done

        stats = {
            "done": total_done,
            "valid": total_valid,
            "pct": round(total_valid / total_done, 4),
        }
        pbar.set_postfix(stats)
        pbar.update(1)

    all_groups = {k: len(file[str(k)]["data"]) for k in sorted(map(int, file.keys()))}
    print(
        f"dataset now has {sum(all_groups.values())} items with {total_done} processed"
    )
