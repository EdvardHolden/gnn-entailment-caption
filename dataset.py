from pathlib import Path
import torch

# from torch.utils.data import DataLoader
from torch_geometric.loader import DataLoader
from torch_geometric.data import Data, InMemoryDataset, Batch, Dataset
from tqdm import tqdm
import os
from typing import List, Sequence
from enum import Enum
import networkx as nx
from itertools import product
import multiprocessing
import pickle
import numpy as np
from sklearn.preprocessing import MinMaxScaler

import config
from graph_parser import graph

from read_problem import read_problem_deepmath, read_problem_tptp


class BenchmarkType(Enum):
    DEEPMATH = "deepmath"
    TPTP = "tptp"

    def __str__(self):
        return self.value

    @classmethod
    def list(cls):
        return list(map(lambda c: c.value, cls))


class LearningTask(Enum):

    PREMISE = "premise"
    SIMILARITY = "similarity"

    def __str__(self):
        return self.value

    @classmethod
    def list(cls):
        return list(map(lambda c: c.value, cls))


def load_ids(id_file) -> List[str]:
    with open(id_file, "r") as f:
        ids = f.readlines()

    return [i.strip() for i in ids]


def _process_problem(
    problem: str,
    problem_dir: str,
    benchmark_type: BenchmarkType,
    remove_argument_node: bool,
    pre_filter=None,
    pre_transform=None,
) -> Data:

    # Read the problem
    conjecture, premises, target = read_problem_from_file(benchmark_type, problem_dir, problem)

    # Construct the data point
    data = construct_graph(conjecture, premises, remove_argument_node=remove_argument_node)
    data.name = Path(problem).stem
    if target is not None:
        data.y = torch.tensor(target)

    if pre_filter is not None:
        data = pre_filter(data)

    if pre_transform is not None:
        data = pre_transform(data)

    return data


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
    edge_index = torch.tensor(np.array([sources, targets]))
    premise_index = torch.tensor(premise_indices)
    conjecture_index = torch.tensor(conjecture_indices)

    data = Data(x=x, edge_index=edge_index, premise_index=premise_index, conjecture_index=conjecture_index)
    return data


class TorchLoadDataset(Dataset):
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

        # Load problem ids
        self.problem_ids = load_ids(id_file)

        # Initialise the super
        super().__init__(self.root.name, transform, pre_transform)

    @property
    def raw_file_names(self) -> List[str]:
        # These are the ids and not really the raw names
        return self.problem_ids

    @property
    def processed_file_names(self) -> List[str]:
        return [f"{self.benchmark_type}_{p}.pt" for p in self.problem_ids]

    def len(self) -> int:
        return len(self.raw_file_names)

    def get(self, idx: int) -> Data:
        data = torch.load(
            os.path.join(self.processed_dir, f"{self.benchmark_type}_{idx}.pt")
        )  # The ids are now the processed names
        return data

    def indices(self) -> Sequence:
        return self.problem_ids

    def process(self):
        for problem in tqdm(self.raw_file_names):

            data = _process_problem(
                problem,
                self.problem_dir,
                self.benchmark_type,
                self.remove_argument_node,
                pre_filter=self.pre_filter,
                pre_transform=self.pre_transform,
            )
            if self.remove_argument_node:
                context = "_remove_arg_node"
            else:
                context = ""

            out_file = Path(self.processed_dir) / f"{self.benchmark_type}{context}_{problem}.pt"
            torch.save(data, out_file)


class TorchMemoryDataset(InMemoryDataset):
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

        # Load problem ids
        self.problem_ids = load_ids(id_file)

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
        if self.remove_argument_node:
            context = "_remove_arg_node"
        else:
            context = ""
        return [f"{self.benchmark_type}{context}_{self.id_partition}.pt"]

    def len(self) -> int:
        return len(self.raw_file_names)

    """
    def get(self, idx: int) -> Data:
        data = torch.load(os.path.join(self.processed_dir, str(idx)))  # The ids are now the processed names
        return data
        """

    def process(self):
        data_list = []

        for problem in tqdm(self.raw_file_names):

            data = _process_problem(
                problem,
                self.problem_dir,
                self.benchmark_type,
                self.remove_argument_node,
                pre_filter=self.pre_filter,
                pre_transform=self.pre_transform,
            )

            # Append the final datapoint to the data list
            data_list.append(data)

        data, slices = self.collate(data_list)
        out = Path(self.processed_dir) / self.processed_file_names[0]
        torch.save((data, slices), out)


class PairData(Data):
    def __init__(
        self,
        edge_index_s: torch.Tensor,
        x_s: torch.Tensor,
        edge_index_t: torch.Tensor,
        x_t: torch.Tensor,
        y: torch.Tensor,
    ):
        super().__init__()
        self.edge_index_s = edge_index_s
        self.x_s = x_s
        self.edge_index_t = edge_index_t
        self.x_t = x_t
        self.y = y

    def __inc__(self, key, value, *args, **kwargs):
        if key == "edge_index_s":
            return self.x_s.size(0)
        if key == "edge_index_t":
            return self.x_t.size(0)
        else:
            return super().__inc__(key, value, *args, **kwargs)


target_min_max_scaler = None  # TODO HACK


def get_pair_dataset(dataset, dataset_path) -> List[PairData]:

    # Get files
    with open(os.path.join(dataset_path, "idx.pkl"), "rb") as f:
        ids = pickle.load(f)

    with open(os.path.join(dataset_path, "target.pkl"), "rb") as f:
        targets = pickle.load(f)
        targets = np.array(targets).reshape(-1, 1)
    assert len(ids) == len(targets)

    # Process targets
    targets = np.sqrt(targets)
    global target_min_max_scaler  #  FIXME quite bad
    if target_min_max_scaler is None:
        target_min_max_scaler = MinMaxScaler().fit(targets)
    targets = target_min_max_scaler.transform(targets)
    targets = targets.reshape(-1)  # Back in expected format

    # Load problem pairs
    pair_list = []
    for target, pair in zip(targets, ids):
        # Get left and right graph in the pair
        if isinstance(dataset, TorchLoadDataset):
            data_s = dataset.get(dataset.problem_ids[int(pair[0])])
            data_t = dataset.get(dataset.problem_ids[int(pair[1])])
        else:
            data_s = dataset.get(pair[0])
            data_t = dataset.get(pair[1])

        # Construct pair data point
        data = PairData(
            edge_index_s=data_s.edge_index,
            x_s=data_s.x,
            edge_index_t=data_t.edge_index,
            x_t=data_t.x,
            y=torch.tensor(target),
        )

        # Add to list
        pair_list += [data]

    return pair_list


def get_data_loader(
    id_file,
    benchmark_type: BenchmarkType = BenchmarkType.DEEPMATH,
    task: LearningTask = LearningTask.PREMISE,
    in_memory: bool = False,
    batch_size: int = config.BATCH_SIZE,
    shuffle: bool = True,
    **kwargs,
) -> DataLoader:

    if in_memory:
        dataset = TorchMemoryDataset(id_file, benchmark_type, **kwargs)
    else:
        dataset = TorchLoadDataset(id_file, benchmark_type, **kwargs)
    kwargs.pop("transform", None)
    kwargs.pop("remove_argument_node", None)

    # Create pair data dataset from loaded problems if set
    if task == LearningTask.SIMILARITY:
        dataset_path = os.path.join(config.dpath, "unsupervised_data/" + Path(id_file).stem)
        dataset = get_pair_dataset(dataset, dataset_path)
        follow_batch = ["x_s", "x_t"]  # Needed for correct sizes
        print("Unsupervised dataset:", len(dataset))
    else:
        follow_batch = None
        print("Dataset:", dataset)

    return DataLoader(
        dataset,
        batch_size=batch_size,
        collate_fn=Batch.from_data_list,
        shuffle=shuffle,
        pin_memory=True,
        follow_batch=follow_batch,
        num_workers=min(multiprocessing.cpu_count() - 1, 8),
        **kwargs,
    )


def test_dataset():
    dataset = TorchLoadDataset("id_files/dev_100.txt", BenchmarkType("deepmath"))
    # dataset = TorchMemoryDataset("id_files/dev_100.txt", BenchmarkType("deepmath"))
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
    print(TorchLoadDataset("id_files/dev_100.txt"))
