from pathlib import Path
from shutil import copyfileobj
from urllib.request import urlopen

import torch
from torch_geometric.data import Data, InMemoryDataset
try:
    from tqdm import tqdm
except ImportError:
    def tqdm(it):
        return it

from logical_entailment_dataset.parser import Parser, propositional_language

VAR, NOT, OR, AND = range(4)
PARSER = Parser(propositional_language())

def download(url, path):
    with urlopen(url) as response, open(path, 'wb') as out:
        copyfileobj(response, out)

def graph_of(line):
    a, b, entailed = line.split(',')[:3]
    entailed = float(entailed)
    text = f"~({a})&({b})"
    result = PARSER.parse(text)

    nodes = []
    node_map = {}
    idx_map = []
    for idx, op, indices in zip(
        range(len(result.ops)),
        result.ops,
        result.inputs
    ):
        indices = tuple(idx_map[index] for index in indices)
        node = (op, indices)

        if op == b'>':
            left, right = indices
            not_left = (b'~', (left,))
            if not_left not in node_map:
                node_map[not_left] = len(nodes)
                nodes.append(not_left)
            left = node_map[not_left]
            node = (b'|', (left, right))

        if node in node_map:
            idx_map.append(node_map[node])
        else:
            node_map[node] = len(nodes)
            idx_map.append(len(nodes))
            nodes.append(node)

    x = []
    sources = []
    targets = []
    for node, indices in nodes:
        current = len(x)
        if node == b'~':
            x.append(NOT)
            sources.append(current)
            targets.append(indices[0])
        elif node == b'|':
            x.append(OR)
            sources.append(current)
            targets.append(indices[0])
            sources.append(current)
            targets.append(indices[1])
        elif node == b'&':
            x.append(AND)
            sources.append(current)
            targets.append(indices[0])
            sources.append(current)
            targets.append(indices[1])
        else:
            assert b'a' <= node and node <= b'z'
            x.append(VAR)

    x = torch.tensor(x)
    edge_index = torch.tensor([sources, targets], dtype=torch.long)
    y = torch.tensor([entailed])
    return Data(x=x, edge_index=edge_index, y=y)

class LogicalEntailmentDataset(InMemoryDataset):
    def __init__(self, root, name='train.txt', transform=None, pre_transform=None):
        self.root = root
        self.name = name
        super().__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return [self.name]

    @property
    def processed_file_names(self):
        stem = Path(self.name).stem
        return [f'{stem}.pt']

    def download(self):
        url = f'https://raw.githubusercontent.com/deepmind/logical-entailment-dataset/master/data/{self.name}'
        out = Path(self.raw_dir) / self.raw_file_names[0]
        print("Downloading...")
        download(url, out)
        print("Done!")

    def process(self):
        with open(self.raw_paths[0], 'r') as f:
            data_list = [graph_of(line) for line in tqdm(f)]
            data, slices = self.collate(data_list)
            out = Path(self.processed_dir) / self.processed_file_names[0]
            torch.save((data, slices), out)
