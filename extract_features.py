import torch
from tqdm import tqdm
from pickle import dump
import os
from pathlib import Path
from sklearn.metrics import pairwise_distances
import numpy as np
import argparse

from model import Model
from common import mk_loader_ltb, mk_loader
import config

parser = argparse.ArgumentParser()
parser.add_argument(
    "--nodes",
    default="all",
    choices=["all", "conjecture", "premise"],
    help="The type of nodes to use in the final embedding",
)
parser.add_argument("--model_path", default="model.pt", help="Path to the model used for embedding")
parser.add_argument("--id_file", default="deepmath.txt", help="Name of the ID file found in id_files/")
parser.add_argument(
    "--print_distances",
    action="store_true",
    default=False,
    help="Prints the euclidean and cosine distance matrix for the computed embedding vectors",
)
parser.add_argument(
    "--library", choices=["tptp", "deepmath"], help="Whether parsing TPTP or Deepmath style problems"
)

# Create hooks
activation = {}


def print_embedding_distances(embeddings):
    feat = []
    for key in sorted(embeddings.keys()):
        feat.append(embeddings[key])

    np.set_printoptions(precision=3)
    np.set_printoptions(suppress=True)

    print("# Euclidean distances")
    print(pairwise_distances(feat, metric="euclidean"))
    print()
    print("# Cosine distances ")
    print(pairwise_distances(feat, metric="cosine"))
    print()


def get_activation(name):
    def hook(model, input, output):
        activation[name] = output.detach()

    return hook


def encode(model, data, nodes=None):
    model.eval()  # Trick to make sure the batchnormalisation does not mess up

    embeddings = {}
    with torch.no_grad():
        for batch in tqdm(data):
            batch = batch.to(config.device)
            if batch.name[0] == "JJT00107+1.p":
                print(batch.x)
            _ = model(batch)

            # Get output of dense layer
            emb = activation["dense"]

            if nodes == "premise":
                # Extract the premise nodes
                emb = emb[batch.premise_index]
            elif nodes == "conjecture":
                emb = emb[batch.conjecture_index]
            elif nodes is None or nodes == "all":
                pass

            # Get the mean of the graph (might change to node/conj indecies?)
            e = torch.mean(emb, 0)
            e = e.detach().numpy()

            # Batch size is 1 for this setup
            embeddings[batch.name[0].strip()] = e
            del batch

    return embeddings


def main():

    args = parser.parse_args()

    # Get set of problems
    if args.library == "tptp":
        # data = mk_loader_ltb("graph_data", args.id_file, batch_size=1, shuffle=False)
        print("HERE")
        # data = mk_loader_ltb("graph_data", args.id_file, caption='/shareddata/home/holden/axiom_caption/generated_problems/mizar_40/sine_1_1/', batch_size=1, shuffle=False)
        data = mk_loader_ltb(
            "graph_data",
            args.id_file,
            caption="/shareddata/home/holden/axiom_caption/generated_problems/mizar_40/sine_1_1/",
            batch_size=1,
            shuffle=False,
        )
    else:
        # data = mk_loader("graph_data", args.id_file, batch_size=1, shuffle=False)
        data = mk_loader(Path(__file__).parent, args.id_file, batch_size=1, shuffle=False)

    print("Number of problems: ", len(data))

    # Load the model
    model = Model(17).to(config.device)
    model.load_state_dict(torch.load(args.model_path))
    # Create hook for getting the intermediate output
    model.dense.register_forward_hook(get_activation("dense"))

    # Compute model embeddings
    print("Computing problem embeddings")
    embeddings = encode(model, data, nodes=args.nodes)

    # Save to path

    # TODO
    res_path = os.path.join(
        "embeddings",
        "graph_features_mizar_merged_sine_1_1" + Path(args.id_file).stem + "_" + args.nodes + ".pkl",
    )
    dump(embeddings, open(res_path, "wb"))

    # Make feature matrix
    if args.print_distances:
        print_embedding_distances(embeddings)


if __name__ == "__main__":
    torch.manual_seed(0)
    main()
