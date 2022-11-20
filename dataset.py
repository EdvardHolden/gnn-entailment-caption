from pathlib import Path
import torch
from torch_geometric.data import Data, InMemoryDataset, Dataset
from tqdm import tqdm
import os
from typing import List

import config
from parser import graph
from utils import read_problem_deepmath, read_problem_tptp


# TODO set mode Mizar vs deepmath

# TODO what s the difference betweent these two?
def construct_graph(conjecture, premises):
    nodes, sources, targets, premise_indices, conjecture_indices = graph(conjecture, premises)
    x = torch.tensor(nodes)
    edge_index = torch.tensor([sources, targets])
    premise_index = torch.tensor(premise_indices)
    conjecture_index = torch.tensor(conjecture_indices)

    data = Data(x=x, edge_index=edge_index, premise_index=premise_index, conjecture_index=conjecture_index)
    return data


# TODO need to add some sort split on deeptmath/merge here!


class TorchDataset(InMemoryDataset):
    def __init__(self, id_file, transform=None, pre_transform=None):
        self.root = Path(__file__).parent
        self.id_file = id_file
        self.id_partition = Path(id_file).stem
        self.problem_dir = config.BENCHMARK_PATHS["deepmath"]

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
        #return [Path(prob).stem + ".pt" for prob in self.problems]
        return [f"{self.id_partition}.pt"]

    def len(self) -> int:
        return len(self.processed_file_names)

    # Need to overwrite this function to operate on the problem names
    def indices(self):
        return self.processed_file_names

    def get(self, idx):
        data = torch.load(os.path.join(self.processed_dir, idx))  # The ids are now the processed names
        return data

    def process(self):
        data_list = []

        # TODO FIX!
        for problem in tqdm(self.raw_file_names):
            # Extract info from the problem
            # conjecture, premises, target = read_problem_deepmath(problem, self.root) TODO
            # conjecture, premises = read_problem_tptp(problem, 'merged_problems')
            # TODO change based on problem: also add path based on problem?
            conjecture, premises = read_problem_tptp(problem, self.problem_dir)
            # Construct the data point
            data = construct_graph(conjecture, premises)
            # Add problem name
            data.name = problem.strip()
            # Add targets
            # data.y = torch.tensor(target)  # TODO

            # Append the final datapoint to the data list
            data_list.append(data)

        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        data, slices = self.collate(data_list)
        out = Path(self.processed_dir) / self.processed_file_names[0]
        torch.save((data, slices), out)


class DeepMathDataset(InMemoryDataset):
    def __init__(self, root, name, transform=None, pre_transform=None):
        self.root = root
        self.name = name
        super().__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return [f"{self.name}"]

    @property
    def processed_file_names(self):
        stem = Path(self.name).stem
        return [f"{stem}.pt"]

    def process(self):
        data_list = []
        with open(self.raw_paths[0], "r") as problems:
            for problem in tqdm(problems):
                # Extract info from the problem
                # conjecture, premises, target = read_problem_deepmath(problem, self.root) TODO
                print(problem)
                # conjecture, premises = read_problem_tptp(problem, 'merged_problems')
                conjecture, premises = read_problem_tptp(
                    problem,
                    "/shareddata/home/holden/axiom_caption/generated_problems/analysis/output_original_unquoted_sine_1_1/",
                )
                # Construct the data point
                data = construct_graph(conjecture, premises)
                # Add problem name
                data.name = problem.strip()
                # Add targets
                # data.y = torch.tensor(target) TODO
                # Append the final datapoint to the data list
                data_list.append(data)
        data, slices = self.collate(data_list)
        out = Path(self.processed_dir) / self.processed_file_names[0]
        torch.save((data, slices), out)


class LTBDataset(Dataset):
    def __init__(self, root, name, caption=None, transform=None, pre_transform=None):

        print("root ", root)
        root = os.path.join(root, name.split(".")[0])  # Make a separate folder for this data
        print("root ", root)
        self.root = root
        self.name = name
        self.caption = caption
        print("CAPTION ", caption)

        # Load problem ids
        with open("id_files/" + self.name, "r") as problems:
            self.problems = [prob.strip() for prob in list(problems)]

        print(self.problems)

        super().__init__(root, transform, pre_transform)

    @property
    def raw_file_names(self):
        return self.problems

    @property
    def processed_file_names(self):
        return [prob.split(".")[0] + ".pt" for prob in self.problems]

    def len(self):
        return len(self.processed_file_names)

    # Need to overwrite this function to operate on the problem names
    def indices(self):
        return self.processed_file_names

    def get(self, idx):
        data = torch.load(os.path.join(self.processed_dir, idx))  # The ids are now the processed names
        return data

    def process(self):

        print("processed_dir ", self.processed_dir)
        print("caption ", self.caption)
        # print(self.problems)

        for problem in tqdm(self.problems):
            # Read the problem caption
            print("pre read: problem ", problem)
            conjecture, axioms = read_problem_tptp(problem, self.caption)
            # Construct the data point
            data = construct_graph(conjecture, axioms)
            # Add problem name
            data.name = problem.strip()

            # Save the data instance
            save_path = os.path.join(self.processed_dir, problem.split(".")[0] + ".pt")
            torch.save(data, save_path)


def test_dataset():
    dataset = TorchDataset("id_files/dev_100.txt")
    print(dataset)


if __name__ == "__main__":

    test_dataset()
