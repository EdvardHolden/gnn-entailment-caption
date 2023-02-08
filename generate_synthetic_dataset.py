import os
import pickle
import argparse
from pathlib import Path
import numpy as np
from tqdm import tqdm
import tensorflow as tf
import pandas as pd
from dataclasses import dataclass
from typing import List, Tuple

from tensorflow import keras
from tensorflow.keras.layers import Embedding
from keras.models import Sequential

import config

physical_devices = tf.config.list_physical_devices("GPU")
for device in physical_devices:
    tf.config.experimental.set_memory_growth(device, True)

import stellargraph as sg
from stellargraph import StellarGraph
import networkx as nx
from tqdm.contrib.concurrent import process_map  # or thread_map

from parser import graph
from read_problem import read_problem_deepmath

# Base names and parameters
IDX_BASE_NAME = "target.pkl"
TARGET_BASE_NAME = "idx.pkl"
WORK_DIR = "unsupervised_data"

embedding_path = "embedding_layer"
if not os.path.exists(embedding_path):
    # Initialise the embedding layer
    node_embedding = Embedding(17, 64)
    # Wrap in a model such that it can be saved
    node_embedding = Sequential(node_embedding)
    node_embedding.save(embedding_path)
else:
    # Load embedding layer
    node_embedding = keras.models.load_model(embedding_path)


def get_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--id_files",
        nargs="+",
        type=str,
        default=["train.txt", "validation.txt", "test.txt"],
        help="List of ID files",
    )

    parser.add_argument(
        "--no_samples",
        nargs="+",
        type=int,
        # TODO using small sizes for testing rn
        default=[100, 50, 10],
        help="Number of samples to generate for each id set",
    )

    parser.add_argument(
        "--problem_dir",
        default=config.PROBLEM_DIR,
        help="Path to the nndata problems",
    )
    parser.add_argument(
        "--max_workers", type=int, default=5, help="Max number of workers to use when computing graph targets"
    )

    parser.add_argument(
        "--recompute",
        action="store_true",
        help="Recomputes the dataset if it already exists",
    )

    return parser


def compute_graph(problem, problem_dir, embed_nodes=True):
    # Read problem and get the problem axioms/conjectures
    conjecture, premises, _ = read_problem_deepmath(problem_dir, problem)

    # Parse the graph
    nodes, sources, targets, premise_indices, conjecture_indices = graph(conjecture, premises)

    # If the problem contains no axioms or no conjecture we do not bother
    if len(premise_indices) == 0 or len(conjecture_indices) == 0:
        return None

    # Need to convert the data into dataframes
    edges = pd.DataFrame({"source": sources, "target": targets})

    # Create the nodes
    if embed_nodes:
        # Embed the node types using an embedding layer
        x = np.array(node_embedding(nodes))
    else:
        # Add nodes with the type id as the feature
        x = pd.DataFrame({"x": nodes})

    # Create graph
    st = StellarGraph(x, edges=edges)

    # Add the indices information
    st.premise_indices = premise_indices
    st.conjecture_indices = conjecture_indices

    return st


def get_problem_ids(id_file):
    with open(os.path.join("id_files", id_file), "r") as f:
        ids = f.readlines()
    ids = [i.strip() for i in ids]
    return ids


def graph_distance(graphs):
    graph1, graph2 = graphs

    spec1 = nx.laplacian_spectrum(graph1.to_networkx(feature_attr=None))
    spec2 = nx.laplacian_spectrum(graph2.to_networkx(feature_attr=None))
    k = min(len(spec1), len(spec2))
    norm = np.linalg.norm(spec1[:k] - spec2[:k])
    return norm


def get_graph_dataset(id_file, problem_dir, embed_nodes=True):

    # Get the graph data
    problems = get_problem_ids(id_file)
    graphs = []
    print("# Processing problems")
    for prob in tqdm(problems):
        g = compute_graph(prob, problem_dir, embed_nodes=embed_nodes)
        # If the graph computation failed we do not add it to the graph set
        if g is not None:
            graphs += [g]
        else:
            # The graph was None so removing the problem name from the list
            problems.remove(prob)
            print(f"Removed problem {prob}")

    # Graph summary
    print(f"Number of graphs: {len(graphs)}")
    summary = pd.DataFrame(
        [(g.number_of_nodes(), g.number_of_edges()) for g in graphs],
        columns=["nodes", "edges"],
    )
    print(summary.describe().round(1))

    # Get the graph generator for the training graphs
    graph_generator = get_graph_generator(graphs)

    return Dataset(graphs, problems, graph_generator)


@dataclass
class Dataset:

    graphs: List
    names: List[str]
    generator: sg.mapper.PaddedGraphSequence


def get_graph_generator(graphs) -> sg.mapper.PaddedGraphSequence:
    return sg.mapper.PaddedGraphGenerator(graphs)


def compute_graph_pair_distances(graphs, no_training_samples, max_workers=5):
    # Make synthetic dataset
    print("Making synthetic dataset")
    graph_idx = np.random.RandomState(0).randint(len(graphs), size=(no_training_samples, 2))

    # Compute targets
    targets = process_map(
        graph_distance, ((graphs[left], graphs[right]) for left, right in graph_idx), max_workers=max_workers
    )

    return graph_idx, targets


def save_dataset(id_file, ids, target_file, targets):
    with open(id_file, "wb") as f:
        pickle.dump(ids, f)
    print(f"Saved file {id_file}")

    with open(target_file, "wb") as f:
        pickle.dump(targets, f)
    print(f"Saved file {target_file}")


def compute_synthetic_dataset(work_dir, recompute_dataset, graph_dataset, no_samples, max_workers):
    # Get the appropriate paths
    id_path, target_path = get_dataset_paths(work_dir)

    # Check if dataset exists and whether it should be overwritten
    if os.path.exists(id_path) and os.path.exists(target_path):
        print("Dataset already exists")
        if not recompute_dataset:
            print("Recompute not set, skipping...")
            return
        else:
            print("Recomputing the dataset")

    print(f"# Computing new dataset of size {no_samples}")
    graph_idx, targets = compute_graph_pair_distances(
        graph_dataset.graphs,
        no_samples,
        max_workers=max_workers,
    )

    # Save the dataset with the given names
    save_dataset(id_path, graph_idx, target_path, targets)


def get_working_directory(path: str) -> str:
    if not os.path.exists(path):
        print("Creating working directory: ", path)
        os.mkdir(path)

    return path


def get_dataset_paths(work_dir: str) -> Tuple[str, str]:
    id_path = os.path.join(work_dir, IDX_BASE_NAME)
    target_path = os.path.join(work_dir, TARGET_BASE_NAME)
    return id_path, target_path


def main():
    # Get arguments
    parser = get_parser()
    args = parser.parse_args()

    # Ensure base directory exists
    get_working_directory(WORK_DIR)

    for id_set, sample_size in zip(args.id_files, args.no_samples):
        print(id_set, sample_size)
        res_dir = get_working_directory(os.path.join(WORK_DIR, Path(id_set).stem))

        # Load graphs
        graph_dataset = get_graph_dataset(id_set, args.problem_dir)

        # Get the synthetic training data set - will be computed if it doesn't already exist
        compute_synthetic_dataset(
            res_dir,
            args.recompute,
            graph_dataset,
            sample_size,
            args.max_workers,
        )

    print("# Finished")


if __name__ == "__main__":
    main()
