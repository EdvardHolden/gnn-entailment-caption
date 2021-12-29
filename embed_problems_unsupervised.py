import os
import pickle

import numpy as np
from tqdm import tqdm
import tensorflow as tf
from tensorflow import keras
import stellargraph as sg
from stellargraph import StellarGraph, StellarDiGraph
import networkx as nx
from tqdm.contrib.concurrent import process_map  # or thread_map
import sys

import pandas as pd

from parser import graph
from dataset import read_problem_tptp


# TODO Undirected or directed?
EPOCHS = 10
NO_TRAINING_SAMPLES = 5


def compute_graph(problem, dir):

    # Read problem and get the problem axioms/conjectures
    conjecture, premises = read_problem_tptp(problem, dir)

    # Parse the graph
    nodes, sources, targets, premise_indices, conjecture_indices = graph(conjecture, premises)

    # Need to convert the data into dataframes
    x = pd.DataFrame({"x": nodes})
    edges = pd.DataFrame({"source": sources, "target": targets})

    # Create the graph structure
    st = StellarGraph(x, edges=edges)

    # Add the indices information
    st.premise_indices = premise_indices
    st.conjecture_indices = conjecture_indices
    #print(premise_indices)
    #print(conjecture_indices)
    #sys.exit(0)

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


def get_graph_dataset(id_file, problem_dir):
    # Get the graph data
    problems = get_problem_ids(id_file)[0:200]  # FIXME TODO
    graphs = []
    print("# Processing problems")
    for prob in tqdm(problems):
        g = compute_graph(prob, problem_dir)
        graphs += [g]

    # Graph summary
    print(f"Number of graphs: {len(graphs)}")
    summary = pd.DataFrame(
        [(g.number_of_nodes(), g.number_of_edges()) for g in graphs],
        columns=["nodes", "edges"],
    )
    print(summary.describe().round(1))

    return graphs, problems


def create_embedding_model(graphs):

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

    # Make synthetic dataset
    print("Making dataset")
    graph_idx = np.random.RandomState(0).randint(len(graphs), size=(NO_TRAINING_SAMPLES, 2))
    # targets = [graph_distance(graphs[left], graphs[right]) for left, right in graph_idx]

    targets = process_map(
        graph_distance, ((graphs[left], graphs[right]) for left, right in graph_idx), max_workers=5
    )

    train_gen = generator.flow(graph_idx, batch_size=10, targets=targets)

    # Train the model
    print("Training the model")
    pair_model.compile(keras.optimizers.Adam(1e-2), loss="mse")
    history = pair_model.fit(train_gen, epochs=EPOCHS, verbose=1)
    sg.utils.plot_history(history)
    return embedding_model, generator


def embed_graphs(file_name, embedding_model, generator, problems, graphs):

    embeddings = embedding_model.predict(generator.flow(graphs))
    print("Embedding")

    # Save the dictionary of embeddings
    result = {}
    for name, emb in zip(problems, embeddings):
        result[name] = emb

    with open(f"{file_name}.pkl", "wb") as f:
        pickle.dump(result, f)


def prune_graph(graph, new_index):
    # Extract subset of nodes
    node_features = graph.node_features()[new_index]

    # Only keep edges in the new subset (offset of 1)
    edges = [(s - 1, t - 1) for s, t in graph.edges() if s - 1 in new_index and t - 1 in new_index]
    print("new_index: ", new_index)

    x = pd.DataFrame(node_features)
    edges = pd.DataFrame(edges, columns=["source", "target"])

    new_graph = StellarGraph(x, edges=edges)
    print("$$")
    print(edges)

    return new_graph


def main(id_file, problem_dir):

    # TODO remember to save the model!

    graphs, problems = get_graph_dataset(id_file, problem_dir)

    '''
    # Train unsupervised embedding model (this takes ages)
    embedding_model, generator = create_embedding_model(graphs)

    # Save the embedding model
    embedding_model.save('embedding_model_save')
    #'''
    embedding_model = keras.models.load_model('embedding_model_save')
    generator = sg.mapper.PaddedGraphGenerator(graphs)
    print(embedding_model)

    # Embed the problems
    """
    print("# Embedding problem")
    embed_graphs("embedding_unsupervised_problem", embedding_model, generator, problems, graphs)
    print("# Embedding conjecture")
    embed_graphs("embedding_unsupervised_conjecture", embedding_model, generator, problems, [prune_graph(g, g.conjecture_indices) for g in graphs])
    """
    print("# Embedding premises")
    embed_graphs("embedding_unsupervised_premises", embedding_model, generator, problems, [prune_graph(g, g.premise_indices) for g in graphs])


def run_main():
    physical_devices = tf.config.list_physical_devices("GPU")
    for device in physical_devices:
        tf.config.experimental.set_memory_growth(device, True)

    problem_dir = "/home/eholden/axiom_caption/data/processed/jjt_sine_1_0/"
    id_file = "axiom_caption_test.txt"
    id_file = "jjt_fof_sine_1_0.txt"

    main(id_file, problem_dir)


if __name__ == "__main__":
    run_main()