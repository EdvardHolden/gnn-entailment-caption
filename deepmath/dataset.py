from pathlib import Path
import torch
from torch_geometric.data import Data, InMemoryDataset
from tqdm import tqdm

from parser import graph

class DeepMathDataset(InMemoryDataset):
    def __init__(self, root, name, transform=None, pre_transform=None):
        self.root = root
        self.name = name
        super().__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return [f'{self.name}']

    @property
    def processed_file_names(self):
        stem = Path(self.name).stem
        return [f'{stem}.pt']

    def read_problem(self, problem):
        path = Path(self.root) / 'nndata' / problem.strip()
        with open(path, 'rb') as f:
            conjecture = next(f).strip()[2:]
            premises, target = zip(*[(
                line[2:],
                1.0 if line.startswith(b'+') else 0.0,
            ) for line in f])
            nodes, sources, targets, premise_indices = graph(
                conjecture,
                premises
            )
            x = torch.tensor(nodes)
            edge_index = torch.tensor([sources, targets])
            premise_index = torch.tensor(premise_indices)
            y = torch.tensor(target)
            data = Data(
                x=x,
                edge_index=edge_index,
                premise_index=premise_index,
                y=y
            )
            return data

    def process(self):
        data_list = []
        with open(self.raw_paths[0], 'r') as problems:
            for problem in tqdm(problems):
                data_list.append(self.read_problem(problem))
        data, slices = self.collate(data_list)
        out = Path(self.processed_dir) / self.processed_file_names[0]
        torch.save((data, slices), out)

if __name__ == '__main__':
    validate = DeepMathDataset(Path(__file__).parent, 'validation.txt')
    print(validate)
    test = DeepMathDataset(Path(__file__).parent, 'test.txt')
    print(test)
    train = DeepMathDataset(Path(__file__).parent, 'train.txt')
    print(train)
