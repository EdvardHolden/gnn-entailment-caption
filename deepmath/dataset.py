from pathlib import Path
import torch
from torch_geometric.data import Data, InMemoryDataset
from tqdm import tqdm
import re
import os

from parser import graph


class DeepMathDataset(InMemoryDataset):
    def __init__(self, root, name, caption=None, transform=None, pre_transform=None):
        self.root = root
        self.name = name
        self.caption = caption
        super().__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return [f'{self.name}']

    @property
    def processed_file_names(self):
        stem = Path(self.name).stem
        return [f'{stem}.pt']

    def _construct_graph(self, conjecture, premises):
        nodes, sources, targets, premise_indices = graph(
            conjecture,
            premises
        )
        x = torch.tensor(nodes)
        edge_index = torch.tensor([sources, targets])
        premise_index = torch.tensor(premise_indices)

        data = Data(x=x, edge_index=edge_index, premise_index=premise_index)
        return data


    def read_problem_tptp(self, problem, problem_dir):

        # HACK: It does not look like there is an actual difference between premises and conjectures in the graph
        # (when we do not need the indicies), so gonna extract the line as the conjecture, adn the rest
        # as the axioms.
        AXIOM_PATTERN = b'((cnf|fof|tff)\\((.*\n)*?.*\\)\\.$)'

        # Change path!
        path = os.path.join(problem_dir, problem.strip())
        print(path)
        with open(path, 'rb') as f:

            text = f.read()
            # Extract all axioms and extract the axiom group
            res = re.findall(AXIOM_PATTERN, text, re.MULTILINE)
            res = [r[0] for r in res]

        # Set first axiom to be the conjecture
        conjecture = res[0]
        premises = tuple(res[1:])

        # Construct the data point
        data = self._construct_graph(conjecture, premises)

        # Add problem name
        data.name = problem.strip()
        # Add targets
        return data


    def read_problem_deepmath(self, problem):
        path = Path(self.root) / 'nndata' / problem.strip()
        with open(path, 'rb') as f:
            conjecture = next(f).strip()[2:]

            premises, target = zip(*[(
                line[2:],
                1.0 if line.startswith(b'+') else 0.0,
            ) for line in f])

        # Construct the data point
        data = self._construct_graph(conjecture, premises)
        # Add problem name
        data.name = problem.strip()
        # Add targets
        data.y = torch.tensor(target)
        return data

    def process(self):
        data_list = []
        with open(self.raw_paths[0], 'r') as problems:
            for problem in tqdm(problems):
                # Load axiom_caption datasets differently than deepmath
                if self.caption is not None:
                    data_list.append(self.read_problem_tptp(problem, self.caption))
                else:
                    data_list.append(self.read_problem_deepmath(problem))
        data, slices = self.collate(data_list)
        out = Path(self.processed_dir) / self.processed_file_names[0]
        torch.save((data, slices), out)


if __name__ == '__main__':

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

    #d = DeepMathDataset(Path(__file__).parent, 'axiom_caption_test.txt', caption='/home/eholden/axiom_caption/data/processed/jjt_sine_1_0/')
    d = DeepMathDataset(
        Path(__file__).parent,
        'axiom_caption_test.txt',
        caption='/home/eholden/JJTProblemFiles/')
    #d = DeepMathDataset(Path(__file__).parent, 'axiom_test.txt', caption='/home/eholden/gnn-entailment-caption/deepmath/nndata/')
    print(d)

