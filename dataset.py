from pathlib import Path
import torch
from torch_geometric.data import Data, InMemoryDataset, Dataset
from tqdm import tqdm
import os
from typing import List
from enum import Enum

import config
from parser import graph
from utils import read_problem_deepmath, read_problem_tptp


def construct_graph(conjecture, premises):
    nodes, sources, targets, premise_indices, conjecture_indices = graph(conjecture, premises)
    x = torch.tensor(nodes)
    edge_index = torch.tensor([sources, targets])
    premise_index = torch.tensor(premise_indices)
    conjecture_index = torch.tensor(conjecture_indices)

    data = Data(x=x, edge_index=edge_index, premise_index=premise_index, conjecture_index=conjecture_index)
    return data


class BenchmarkType(Enum):
    DEEPMATH = "deepmath"
    TPTP = "tptp"

    def __str__(self):
        return self.value


class TorchDataset(InMemoryDataset):
    def __init__(self, id_file, benchmark_type, transform=None, pre_transform=None):
        self.root = Path(__file__).parent
        self.id_file = id_file
        self.id_partition = Path(id_file).stem
        self.benchmark_type = benchmark_type
        self.problem_dir = config.BENCHMARK_PATHS[str(benchmark_type)]

        # Load problem ids and store in list
        with open(self.id_file, "r") as problems:
            self.problems = [prob.strip() for prob in list(problems)]

        # Initialise the super
        super().__init__(self.root.name, transform, pre_transform)

        # Start process of getting the data
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self) -> List[str]:
        # These are the ids and not really the raw names
        return self.problems

    @property
    def processed_file_names(self) -> List[str]:
        # return [Path(prob).stem + ".pt" for prob in self.problems]
        return [f"{self.id_partition}.pt"]

    def len(self) -> int:
        return len(self.raw_file_names)

    def get(self, idx):
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


def test_dataset():
    dataset = TorchDataset("id_files/dev_100.txt", BenchmarkType("deepmath"))
    print(dataset)
    print(len(dataset))


if __name__ == "__main__":

    test_dataset()
