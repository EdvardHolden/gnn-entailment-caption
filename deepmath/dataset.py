from pathlib import Path
import torch
from torch_geometric.data import Data, InMemoryDataset, Dataset
from tqdm import tqdm
import re
import os

from parser import graph

CLAUSE_PATTERN = b'((cnf|fof|tff)\\((.*\n)*?.*\\)\\.$)'


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
        # as the axioms. TODO fix

        # Change path!
        path = os.path.join(problem_dir, problem.strip())
        with open(path, 'rb') as f:

            text = f.read()
            # Extract all axioms and extract the axiom group
            res = re.findall(CLAUSE_PATTERN, text, re.MULTILINE)
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

# TODO make new class
# TODO update_processed_file_names
# TODO update process with save method
# TODO implement len
# TODO implement get


# ##################################################################################################33

#class LTBDataset(Dataset):
class LTBDataset(InMemoryDataset):

    def __init__(self, root, name, caption, transform=None, pre_transform=None):
        self.root = root
        self.name = name
        self.caption = caption

        # Load problem ids
        #with open(self.raw_paths[0], 'r') as problems:
        with open('raw/' + self.name, 'r') as problems:
            self.problems = [prob.strip() for prob in list(problems)]


        super().__init__(root, transform, pre_transform)
        #self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return self.problems

    @property
    def processed_file_names(self):
        return [prob.split('.')[0] + '.pt' for prob in self.problems]

    def len(self):
        return len(self.processed_file_names)

    def get(self, idx):
        data = torch.load(os.path.join(self.processed_dir, idx.split('.')[0] + '.pt'))
        return data

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

    def _get_clauses(self, problem_dir, problem):
        # Read the problem file
        path = os.path.join(problem_dir, problem.strip())
        with open(path, 'rb') as f:
            text = f.read()

        print(problem)

        # Extract all axioms and extract the axiom group
        res = re.findall(CLAUSE_PATTERN, text, re.MULTILINE)
        res = [r[0] for r in res]

        # Convert all cnf, tff clauses to fof (will remove type information later)
        for n in range(len(res)):
            res[n] = res[n].replace(b'cnf(', b'fof(')
            res[n] = res[n].replace(b'tff(', b'fof(')

        return res

    def _split_clauses(self, clauses):
        # TODO need to split it and extract properly!
        # Filter all types from tff!
        axioms = []
        conjecture = None # TODO convert to list

        for clause in clauses:
            # TODO if type, discard
            if b'conjecture' in clause:
                conjecture = clause
            elif b'type' in clause:
                pass # We discard tff types
            else:
                axioms += [clause]

        return conjecture, tuple(axioms)


    def read_problem_tptp(self, problem, problem_dir):

        # Extract the clauses from the problem
        clauses = self._get_clauses(problem_dir, problem)

        # Set first axiom to be the conjecture
        # TODO convert to multiple conjectures later!
        conjecture, axioms = self._split_clauses(clauses)

        # Construct the data point
        data = self._construct_graph(conjecture, axioms)

        # Add problem name
        data.name = problem.strip()
        # Add targets
        return data


    def process(self):

        for problem in tqdm(self.problems):
            # Read the problem caption
            data = self.read_problem_tptp(problem, self.caption)
            # Save the data instance
            torch.save(data, os.path.join(self.processed_dir, problem.split('.')[0] + '.pt'))



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

    # TODO update root (first argument, to be more specific!)
    #d = LTBDataset(Path(__file__).parent, 'axiom_caption_test.txt', caption='/home/eholden/axiom_caption/data/processed/jjt_sine_1_0/')
    #d = LTBDataset(Path(__file__).parent, 'jjt_sine_1_0.txt', caption='/home/eholden/axiom_caption/data/processed/jjt_sine_1_0/')
    d = LTBDataset(Path(__file__).parent, 'jjt_fof_sine_1_0.txt', '/home/eholden/axiom_caption/data/processed/jjt_sine_1_0/')

    #d = LTBDataset(Path(__file__).parent, 'axiom_caption_test.txt', caption='/home/eholden/JJTProblemFiles/')

    #d = LTBDataset(Path(__file__).parent, 'jjt_fof.txt', caption='/home/eholden/JJTProblemFiles/')
    #d = LTBDataset(Path(__file__).parent, 'jjt_fof_half.txt', caption='/home/eholden/JJTProblemFiles/')
    #d = DeepMathDataset(Path(__file__).parent, 'jjt_fof_other.txt', caption='/home/eholden/JJTProblemFiles/')
    #d = DeepMathDataset(Path(__file__).parent, 'axiom_test.txt', caption='/home/eholden/gnn-entailment-caption/deepmath/nndata/')
    print(d)
    #print(d.raw_file_names)
    #print(d.processed_file_names)

