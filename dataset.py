from pathlib import Path
import torch
from torch_geometric.data import Data, InMemoryDataset, Dataset
from tqdm import tqdm
import os

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
                conjecture, premises, target = read_problem_deepmath(problem, self.root)
                # Construct the data point
                data = construct_graph([conjecture], premises)
                # Add problem name
                data.name = problem.strip()
                # Add targets
                data.y = torch.tensor(target)
                # Append the final datapoint to the data list
                data_list.append(data)
        data, slices = self.collate(data_list)
        out = Path(self.processed_dir) / self.processed_file_names[0]
        torch.save((data, slices), out)


class LTBDataset(Dataset):
    def __init__(self, root, name, caption=None, transform=None, pre_transform=None):

        root = os.path.join(root, name.split(".")[0])  # Make a separate folder for this data
        self.root = root
        self.name = name
        self.caption = caption

        # Load problem ids
        with open("raw/" + self.name, "r") as problems:
            self.problems = [prob.strip() for prob in list(problems)]

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

        for problem in tqdm(self.problems):
            # Read the problem caption
            conjecture, axioms = read_problem_tptp(problem, self.caption)
            # Construct the data point
            data = construct_graph(conjecture, axioms)
            # Add problem name
            data.name = problem.strip()

            # Save the data instance
            save_path = os.path.join(self.processed_dir, problem.split(".")[0] + ".pt")
            torch.save(data, save_path)


if __name__ == "__main__":

    # TODO clean up this mess!
    """
    print(Path(__file__).parent)
    print("# Validation")
    validate = DeepMathDataset(Path(__file__).parent, 'validation.txt')
    print(validate)
    #"""
    """
    print("# Testing")
    test = DeepMathDataset(Path(__file__).parent, 'test.txt')
    print(test)
    print("# Training")
    train = DeepMathDataset(Path(__file__).parent, 'train.txt')
    print(train)
    """

    d = LTBDataset(
        "graph_data",
        "axiom_caption_test.txt",
        caption="/home/eholden/axiom_caption/data/processed/jjt_sine_1_0/",
    )
    # d = LTBDataset(Path(__file__).parent, 'jjt_fof_sine_1_0.txt', caption='/home/eholden/axiom_caption/data/processed/jjt_sine_1_0/')
    """
    d = LTBDataset(
        'graph_data',
        'jjt_fof_sine_1_0.txt',
        caption='/home/eholden/axiom_caption/data/processed/jjt_sine_1_0/')
    #print(d.get('JJT00001+1'))
    #"""

    print(d)
    # print(d.raw_file_names)
    # print(d.processed_file_names)
