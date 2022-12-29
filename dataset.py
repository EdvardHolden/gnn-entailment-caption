from pathlib import Path
import torch
from torch.utils.data import DataLoader
from torch_geometric.data import Data, InMemoryDataset, Batch, Dataset
from tqdm import tqdm
import os
from typing import List, Sequence
from enum import Enum
import networkx as nx
from itertools import product

import config
from graph_parser import graph

from read_problem import read_problem_deepmath, read_problem_tptp


class BenchmarkType(Enum):
    DEEPMATH = "deepmath"
    TPTP = "tptp"

    def __str__(self):
        return self.value


def load_ids(id_file) -> List[str]:
    with open(id_file, "r") as f:
        ids = f.readlines()

    return [i.strip() for i in ids]


def read_problem_from_file(benchmark_type, problem_dir, problem):
    target = None
    if benchmark_type is BenchmarkType.DEEPMATH:
        conjecture, premises, target = read_problem_deepmath(problem_dir, problem)
    elif benchmark_type is BenchmarkType.TPTP:
        conjecture, premises = read_problem_tptp(problem_dir, problem)
    else:
        raise ValueError(f"Not implemented problem loader for benchmark {benchmark_type}")

    return conjecture, premises, target


def remove_node_type(nodes, sources, targets, premise_indices, conjecture_indices, node_type=4):
    # Check if node type exists in the graph
    if node_type not in nodes:
        return nodes, sources, targets, premise_indices, conjecture_indices

    # Convert into a networkx Directional graph
    G = nx.DiGraph()
    G.add_edges_from(list(zip(sources, targets)))

    # Transfer the node types
    attr = {i: t for i, t in enumerate(nodes)}
    nx.set_node_attributes(G, attr, name="type")

    # Remove the given node type
    for node_id in list(G.nodes):

        # Check if node is of the correct type
        if G.nodes[node_id]["type"] != node_type:
            continue

        # Get all in/out edges and remap current node
        in_nodes = [a for a, b in G.in_edges(node_id)]
        out_nodes = [b for a, b in G.out_edges(node_id)]

        new_edges = list(product(*[in_nodes, out_nodes]))
        G.add_edges_from(new_edges)

        # Finally remove the node
        G.remove_node(node_id)

    # Remap and restructure indices to get it back to the original format
    node_map = {i: n for n, i in enumerate(list(G.nodes))}
    map_node = {n: i for n, i in enumerate(list(G.nodes))}

    new_targets = [t for s, t in G.edges]
    new_targets = list(map(node_map.get, new_targets))

    new_sources = [s for s, t in G.edges]
    new_sources = list(map(node_map.get, new_sources))

    new_nodes = [G.nodes[map_node[n]]["type"] for n in range(len(map_node))]

    new_premise_indices = list(map(node_map.get, premise_indices))
    new_conjecture_indices = list(map(node_map.get, conjecture_indices))

    return new_nodes, new_sources, new_targets, new_premise_indices, new_conjecture_indices


def construct_graph(conjecture: List[str], premises: List[str], remove_argument_node=False) -> Data:
    # Parse the formulae into a graph
    nodes, sources, targets, premise_indices, conjecture_indices = graph(conjecture, premises)

    if remove_argument_node:
        nodes, sources, targets, premise_indices, conjecture_indices = remove_node_type(
            nodes, sources, targets, premise_indices, conjecture_indices
        )

    x = torch.tensor(nodes)
    edge_index = torch.tensor([sources, targets])
    premise_index = torch.tensor(premise_indices)
    conjecture_index = torch.tensor(conjecture_indices)

    data = Data(x=x, edge_index=edge_index, premise_index=premise_index, conjecture_index=conjecture_index)
    return data


class TorchDatasetNEW(Dataset):
    def __init__(
        self,
        id_file: str,
        benchmark_type: BenchmarkType = BenchmarkType.DEEPMATH,
        transform=None,
        pre_transform=None,
        remove_argument_node: bool = False,
    ):
        self.root = Path(".")
        self.id_file = id_file
        self.id_partition = Path(id_file).stem
        self.benchmark_type = benchmark_type
        self.problem_dir = config.BENCHMARK_PATHS[str(benchmark_type)]
        self.remove_argument_node = remove_argument_node

        # Load problem ids and store in list
        with open(self.id_file, "r") as problems:
            self.problem_ids = [prob.strip() for prob in list(problems)]

        # Initialise the super
        super().__init__(self.root.name, transform, pre_transform)

    @property
    def raw_file_names(self) -> List[str]:
        # These are the ids and not really the raw names
        return self.problem_ids

    @property
    def processed_file_names(self) -> List[str]:
        # return [Path(prob).stem + ".pt" for prob in self.problems]
        return [f"{self.benchmark_type}_{p}.pt" for p in self.problem_ids]

    def len(self) -> int:
        return len(self.raw_file_names)

    def get(self, idx) -> Data:
        data = torch.load(
            os.path.join(self.processed_dir, f"{self.benchmark_type}_{idx}.pt")
        )  # The ids are now the processed names
        return data

    def indices(self) -> Sequence:
        # return range(self.len()) if self._indices is None else self._indices
        return self.problem_ids

    def process(self):

        for problem in tqdm(self.raw_file_names):

            # Read the problem
            conjecture, premises, target = read_problem_from_file(
                self.benchmark_type, self.problem_dir, problem
            )

            # Construct the data point
            data = construct_graph(conjecture, premises, remove_argument_node=self.remove_argument_node)
            data.name = Path(problem).stem
            if target is not None:
                data.y = torch.tensor(target)

            if self.pre_filter is not None:
                data = self.pre_filter(data)

            if self.pre_transform is not None:
                data = self.pre_transform(data)

            # data, slices = self.collate(data)
            out = Path(self.processed_dir) / f"{self.benchmark_type}_{problem}.pt"
            torch.save(data, out)


"""
class TorchDataset(InMemoryDataset):
    def __init__(self, id_file: str, benchmark_type: BenchmarkType = BenchmarkType.DEEPMATH, transform=None,
                 pre_transform=None):
        self.root = Path(__file__).parent
        self.id_file = id_file
        self.id_partition = Path(id_file).stem
        self.benchmark_type = benchmark_type
        self.problem_dir = config.BENCHMARK_PATHS[str(benchmark_type)]

        # Load problem ids and store in list
        with open(self.id_file, "r") as problems:
            self.problem_ids = [prob.strip() for prob in list(problems)]

        # Initialise the super
        super().__init__(self.root.name, transform, pre_transform)

        # Start process of getting the data
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self) -> List[str]:
        # These are the ids and not really the raw names
        return self.problem_ids

    @property
    def processed_file_names(self) -> List[str]:
        # return [Path(prob).stem + ".pt" for prob in self.problems]
        return [f"{self.benchmark_type}_{self.id_partition}.pt"]

    def len(self) -> int:
        return len(self.raw_file_names)

    def get(self, idx) -> Data:
        data = torch.load(os.path.join(self.processed_dir, idx))  # The ids are now the processed names
        return data

    def process(self):
        data_list = []

        for problem in tqdm(self.raw_file_names):
            target = None

            # Read the problem
            if self.benchmark_type is BenchmarkType.DEEPMATH:
                conjecture, premises, target = read_problem_deepmath(self.problem_dir, problem)
            elif self.benchmark_type is BenchmarkType.TPTP:
                conjecture, premises = read_problem_tptp(self.problem_dir, problem)
            else:
                raise ValueError(f"Not implemented problem loader for benchmark {self.benchmark_type}")

            # Construct the data point
            data = construct_graph(conjecture, premises)
            data.name = Path(problem).stem
            if target is not None:
                data.y = torch.tensor(target)

            # Append the final datapoint to the data list
            data_list.append(data)

        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        data, slices = self.collate(data_list)
        out = Path(self.processed_dir) / self.processed_file_names[0]
        torch.save((data, slices), out)
        
"""


def get_data_loader(
    id_file,
    benchmark_type: BenchmarkType = BenchmarkType.DEEPMATH,
    batch_size: int = config.BATCH_SIZE,
    shuffle: bool = True,
    remove_argument_node: bool = False,
    **kwargs,
):
    dataset = TorchDatasetNEW(id_file, benchmark_type, remove_argument_node=remove_argument_node)
    print("Dataset:", dataset)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        collate_fn=Batch.from_data_list,
        shuffle=shuffle,
        pin_memory=True,
        num_workers=8,
        **kwargs,
    )


def test_dataset():
    dataset = TorchDatasetNEW("id_files/dev_100.txt", BenchmarkType("deepmath"))
    print(dataset)
    print(len(dataset))


def test_data_loader():
    loader = get_data_loader("id_files/dev_100.txt", BenchmarkType("deepmath"))
    print(loader)


if __name__ == "__main__":
    print(1)
    test_dataset()
    print(2)
    test_data_loader()
    print(3)
    print(TorchDatasetNEW("id_files/dev_100.txt"))
