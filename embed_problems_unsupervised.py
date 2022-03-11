import os
import pickle
import argparse
from pathlib import Path
import numpy as np
from tqdm import tqdm
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Embedding, Input
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
parser.add_argument(
    "--evaluate",
    default=False,
    action="store_true",
    help="Evaluate the model on the training and validation set",
)

# Set dataset files base names. e.g. 0: train, 1: 6000
IDX_BASE_NAME = "target.pkl"
TARGET_BASE_NAME = "idx.pkl"
MODEL_DIR_BASE_NAME = "embedding_model"


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


def get_graph_generator(graphs):
    return sg.mapper.PaddedGraphGenerator(graphs)


def initialise_embedding_model(generator):

    # TODO  pooling (callable, optional): a Keras layer or function that takes two arguments and return
    gc_model = sg.layer.GCNSupervisedGraphClassification(
        [64, 32], ["relu", "relu"], generator, pool_all_layers=True  # TODO what about the pool argument?
    )
    inp1, out1 = gc_model.in_out_tensors()
    inp2, out2 = gc_model.in_out_tensors()
    embedding_model = keras.Model(inp1, out1)

    return embedding_model


def _get_in_out_tensor(generator, embedding_model):
    """Use this function to get the list of input tensor used in the gc model.
    This makes it possible to use TODO"""

    # x_t = Input(shape=(None, generator.node_features_size))
    x_t = Input(shape=(None, 1))  # TODO Need to retrain and set to above!
    mask = Input(shape=(None,), dtype=tf.bool)
    A_m = Input(shape=(None, None))

    x_inp = [x_t, mask, A_m]
    x_out = embedding_model(x_inp)

    return x_inp, x_out


def get_pair_model(generator, embedding_model):

    inp1, out1 = _get_in_out_tensor(generator, embedding_model)
    inp2, out2 = _get_in_out_tensor(generator, embedding_model)

    vec_distance = tf.norm(out1 - out2, axis=1)
    pair_model = keras.Model(inp1 + inp2, vec_distance)

    return pair_model


def initialise_embedding_model_old(graphs):
    # TODO rewrite/remove?

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

    # TODO it would be nice to have more of a direcotry structure for this.
    # e.g.
    # unsupervised_6000_train/
    # target.pkl
    # idx.pkl
    # saved_model/
    # embeddings___

    # Get arguments
    args = parser.parse_args()

    # Check working direcotry for storing models, datasets and embeddings
    work_dir = f"unsupervised_{Path(args.train_id_file).stem}_{args.no_training_samples}"
    if not os.path.exists(work_dir):
        print("Creating working directory: ", work_dir)
        os.mkdir(work_dir)

    # Load training data
    problem_graphs, problem_names = get_graph_dataset(args.train_id_file, args.problem_dir)

    # If flag is set to recompute dataset or the appropriate files do not exist, we recompute the dataset
    id_path = os.path.join(work_dir, IDX_BASE_NAME)
    target_path = os.path.join(work_dir, TARGET_BASE_NAME)
    if args.recompute_dataset or not (os.path.exists(id_path) and os.path.exists(target_path)):
        print(f"# Computing new dataset of size {args.no_training_samples}")
        graph_idx, targets = compute_dataset(
            problem_graphs,
            args.no_training_samples,
            id_path,
            target_path,
            max_workers=args.max_workers,
        )
    else:
        print("# Loading existing dataset")
        graph_idx, targets = load_dataset(id_path, target_path)

    #  Scale the targets
    graph_idx, targets = process_dataset(graph_idx, targets)

    # Get the graph generator
    graph_generator = get_graph_generator(problem_graphs)

    # Get the name of the appropriate model dir
    model_dir = os.path.join(work_dir, MODEL_DIR_BASE_NAME)

    # As default we have no pair model
    pair_model = None

    # Train the model if retraining flag is set or it does not already exist
    if args.retrain or not os.path.exists(model_dir):
        # Initialise the embedding and pair model
        embedding_model = initialise_embedding_model(graph_generator)
        pair_model = get_pair_model(graph_generator, embedding_model)

        # Train the pair model
        print("Training embedding model on the pair dataset")
        train_pair_model(pair_model, graph_generator, graph_idx, targets, args.epochs)

        # Save the embedding model
        print("Saving the embedding model to ", model_dir)
        embedding_model.save(model_dir)
    else:
        print("Loading existing model")
        embedding_model = keras.models.load_model(model_dir)

    if args.evaluate:
        # Create pair model if it doesnt exist due to loading existing embedding model
        if pair_model is None:
            pair_model = get_pair_model(graph_generator, embedding_model)

        print("Evaluating pair model")
        evaluate_pair_model(pair_model, evaluation_dataset)

    # Embed the problems
    if args.embed_problems:
        embed_problems(
            embedding_model,
            graph_generator,
            problem_names,
            problem_graphs,
            "{args.train_id_file}_{args.no_training_samples}",  # Use context parameters as file infix
        )
    print("# Finished")


def run_main():

    main()


if __name__ == "__main__":
    run_main()
