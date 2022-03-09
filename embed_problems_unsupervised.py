import os
import pickle
import argparse
from pathlib import Path
import numpy as np
from tqdm import tqdm
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Embedding
import pandas as pd
from sklearn.preprocessing import StandardScaler

physical_devices = tf.config.list_physical_devices("GPU")
for device in physical_devices:
    tf.config.experimental.set_memory_growth(device, True)

from keras.models import Sequential

import stellargraph as sg
from stellargraph import StellarGraph
import networkx as nx
from tqdm.contrib.concurrent import process_map  # or thread_map

from parser import graph
from utils import read_problem_deepmath

parser = argparse.ArgumentParser()
parser.add_argument(
    "--train_id_file", default="train.txt", help="Name of the file containing the training data in raw/"
)
parser.add_argument(
    "--problem_dir", default="/home/eholden/gnn-entailment-caption/", help="Path to the nndata problems"
)
parser.add_argument(
    "--max_workers", type=int, default=5, help="Max number of workers to use when computing graph targets"
)

# TODO add model types as well?
# TODO What about the validation set?
parser.add_argument(
    "--no_training_samples",
    default=6000,
    type=int,
    help="Number of training pair-samples to compute from the training set",
)
parser.add_argument("--epochs", default=200, type=int, help="The number of epochs to train the model for")
parser.add_argument(
    "--recompute_dataset",
    default=False,
    action="store_true",
    help="If set we recompute the graph dataset for the given parameters",
)
parser.add_argument("--retrain", default=False, action="store_true", help="Retrain existing model")
parser.add_argument(
    "--embed_problems", default=False, action="store_true", help="Use the GNN to embed the problems"
)

# Set dataset files base names. e.g. 0: train, 1: 6000
IDX_BASE_NAME = "graph_{0}_dataset_{1}_target.pkl"
TARGET_BASE_NAME = "graph_{0}_dataset_{1}_idx.pkl"
MODEL_DIR_BASE_NAME = "embedding_model_{0}_{1}_save"


# For the different axiom types we manually create an embedding layer for mapping
# the types to the embedding space. This creates or loads and existing embedding layer
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


def compute_graph(problem, problem_dir, embed_nodes=True):

    # Read problem and get the problem axioms/conjectures
    # conjecture, premises = read_problem_tptp(problem, dir)
    conjecture, premises, _ = read_problem_deepmath(problem, problem_dir)

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

    with open(os.path.join("raw", id_file), "r") as f:
        ids = f.readlines()
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

    return graphs, problems


def initialise_embedding_model(graphs):

    # Initialize the graph model
    generator = sg.mapper.PaddedGraphGenerator(graphs)
    gc_model = sg.layer.GCNSupervisedGraphClassification(
        [64, 32], ["relu", "relu"], generator, pool_all_layers=True
    )
    inp1, out1 = gc_model.in_out_tensors()
    inp2, out2 = gc_model.in_out_tensors()

    vec_distance = tf.norm(out1 - out2, axis=1)
    pair_model = keras.Model(inp1 + inp2, vec_distance)
    embedding_model = keras.Model(inp1, out1)

    return embedding_model, pair_model, generator


def compute_dataset(graphs, no_training_samples, id_file_name, target_file_name, max_workers=5):

    # Make synthetic dataset
    print("Making synthetic dataset")
    graph_idx = np.random.RandomState(0).randint(len(graphs), size=(no_training_samples, 2))

    # Compute targets
    targets = process_map(
        graph_distance, ((graphs[left], graphs[right]) for left, right in graph_idx), max_workers=max_workers
    )

    # Save the dataset with the given names
    save_dataset(id_file_name, graph_idx, target_file_name, targets)

    return graph_idx, targets


def train_pair_model(pair_model, generator, graph_idx, targets, epochs):

    train_gen = generator.flow(graph_idx, batch_size=10, targets=targets)

    # Train the model
    print("Training the model")
    pair_model.compile(keras.optimizers.Adam(1e-2), loss="mse")
    history = pair_model.fit(train_gen, epochs=epochs, verbose=1)
    sg.utils.plot_history(history)


def embed_graphs(file_name, embedding_model, generator, problems, graphs):

    embeddings = embedding_model.predict(generator.flow(graphs))
    print("Embedding")

    # Save the dictionary of embeddings
    result = {}
    for name, emb in zip(problems, embeddings):
        result[name] = emb

    print("Saving embedding file to: ", file_name)
    with open(f"{file_name}.pkl", "wb") as f:
        pickle.dump(result, f)


def prune_graph(graph, new_index):
    # Extract subset of nodes
    node_features = graph.node_features()[new_index]

    # Only keep edges in the new subset (offset of 1)
    edges = [(s - 1, t - 1) for s, t in graph.edges() if s - 1 in new_index and t - 1 in new_index]

    x = pd.DataFrame(node_features, index=new_index)
    edges = pd.DataFrame(edges, columns=["source", "target"])

    new_graph = StellarGraph(x, edges=edges)

    return new_graph


def load_dataset(id_file, target_file):
    print(f"Loading file {id_file}")
    with open(id_file, "rb") as f:
        idx = pickle.load(f)

    print(f"Loading file {target_file}")
    with open(target_file, "rb") as f:
        targets = pickle.load(f)

    return idx, targets


def save_dataset(id_file, ids, target_file, targets):

    with open(id_file, "wb") as f:
        pickle.dump(ids, f)
    print(f"Saved file {id_file}")

    with open(target_file, "wb") as f:
        pickle.dump(targets, f)
    print(f"Saved file {target_file}")


def embed_problems(embedding_model, generator, problem_names, problem_graphs, file_name_infix):

    # Compute file name prefixes
    file_prefix = "embedding_unsupervised_" + file_name_infix

    print("# Embedding problem")
    embed_graphs(file_prefix + "_problem", embedding_model, generator, problem_names, problem_graphs)

    # FIXME this is somewhat pointless?
    print("# Embedding conjecture")
    embed_graphs(
        file_prefix + "_conjecture",
        embedding_model,
        generator,
        problem_names,
        [prune_graph(g, g.conjecture_indices) for g in problem_graphs],
    )

    print("# Embedding premises")
    embed_graphs(
        file_prefix + "_premises",
        embedding_model,
        generator,
        problem_names,
        [prune_graph(g, g.premise_indices) for g in problem_graphs],
    )


def process_dataset(idx, targets):

    # Scale the targets
    scaler = StandardScaler()
    targets = scaler.fit_transform(targets)

    return idx, targets


def main():

    # Get arguments
    args = parser.parse_args()
    # Load training data
    problem_graphs, problem_names = get_graph_dataset(args.train_id_file, args.problem_dir)

    # Compute the file names for the dataset
    id_name = Path(args.train_id_file).stem
    train_file_id = IDX_BASE_NAME.format(id_name, args.no_training_samples)
    train_file_targets = TARGET_BASE_NAME.format(id_name, args.no_training_samples)

    # If flag is set to recompute dataset or the appropriate files do not exist, we recompute the dataset
    if args.recompute_dataset or not (os.path.exists(train_file_id) and os.path.exists(train_file_targets)):
        print(f"# Computing new dataset of size {args.no_training_samples}")
        graph_idx, targets = compute_dataset(
            problem_graphs,
            args.no_training_samples,
            train_file_id,
            train_file_targets,
            max_workers=args.max_workers,
        )
    else:
        print("# Loading existing dataset")
        graph_idx, targets = load_dataset(train_file_id, train_file_targets)

    graph_idx, targets = process_dataset(graph_idx, targets)

    # Create the models and data generator
    embedding_model, pair_model, generator = initialise_embedding_model(problem_graphs)

    # Get the name of the appropriate model dir
    model_dir = MODEL_DIR_BASE_NAME.format(id_name, args.no_training_samples)

    # Train the model if retraining flag is set or it does not already exist
    if args.retrain or not os.path.exists(model_dir):
        print("Training embedding model on the pair dataset")

        # Train the pair model
        train_pair_model(pair_model, generator, graph_idx, targets, args.epochs)

        # Save the embedding model
        print("Saving the embedding model to ", model_dir)
        embedding_model.save(model_dir)
    else:
        print("Loading model")
        embedding_model = keras.models.load_model(model_dir)
        generator = sg.mapper.PaddedGraphGenerator(problem_graphs)
        # TODO due to the evaluation I want to get the pair_model with the pre trained gnn as well
    print(embedding_model)

    # Embed the problems
    if args.embed_problems:
        embed_problems(
            embedding_model,
            generator,
            problem_names,
            problem_graphs,
            "{args.train_id_file}_{args.no_training_samples}",  # Use context parameters as file infix
        )
    print("# Finished")


def run_main():

    main()


if __name__ == "__main__":
    run_main()
