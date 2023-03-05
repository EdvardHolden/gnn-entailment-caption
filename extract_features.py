import torch
from tqdm import tqdm
from pickle import dump
import os
from pathlib import Path
from sklearn.metrics import pairwise_distances
import numpy as np
import argparse


from dataset import get_data_loader, BenchmarkType, load_graph_params, LearningTask
import config
from model import load_model


def get_extraction_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()

    parser.add_argument("model_dir", help="Path to the model used for embedding")
    parser.add_argument(
        "learning_task",
        # default=LearningTask.PREMISE,
        choices=list(LearningTask),
        type=lambda x: LearningTask(x),
        help="Learning task of the model (Must match the original training objective)",
    )
    parser.add_argument("--id_file", default="deepmath.txt", help="Name of the ID file found in id_files/")

    parser.add_argument(
        "--nodes",
        default="all",
        choices=["all", "conjecture", "premise"],
        help="The type of nodes to use in the final embedding",
    )
    parser.add_argument("--result_infix_path", default="")

    parser.add_argument(
        "--print_distances",
        action="store_true",
        default=False,
        help="Prints the euclidean and cosine distance matrix for the computed embedding vectors",
    )
    parser.add_argument(
        "--benchmark_type",
        default=BenchmarkType.DEEPMATH,
        choices=list(BenchmarkType),
        type=lambda x: BenchmarkType(x),
        help="Benchmark type of the problems.",
    )
    parser.add_argument(
        "--in_memory", action="store_true", help="Set dataset to in memory (may not always work)"
    )

    return parser


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


def encode(model, data, nodes=None):
    model.eval()  # Trick to make sure the batch-normalisation does not mess up

    embeddings = {}
    with torch.no_grad():
        for batch in tqdm(data):
            batch = batch.to(config.device)
            if batch.name[0] == "JJT00107+1.p":
                print(batch.x)
            emb, _ = model(batch)

            if nodes == "premise":
                # Extract the premise nodes
                emb = emb[batch.premise_index]
            elif nodes == "conjecture":
                emb = emb[batch.conjecture_index]
            elif nodes is None or nodes == "all":
                pass

            # Get the mean of the embeddings
            e = torch.mean(emb, 0)
            e = e.detach().numpy()

            # Batch size is 1 for this setup
            embeddings[batch.name[0].strip()] = e
            del batch

    return embeddings


def main():
    parser = get_extraction_parser()
    args = parser.parse_args()

    # Load the model
    model = load_model(args.model_dir, args.learning_task)
    model = model.to(config.device)
    model.eval()

    # Get data loader
    graph_params = load_graph_params(args.model_dir)
    graph_data = get_data_loader(
        args.id_file,
        args.benchmark_type,
        batch_size=1,
        shuffle=False,
        task=args.learning_task,
        in_memory=args.in_memory,
        **graph_params
    )
    print("Number of problems: ", len(graph_data))

    # Compute model embeddings
    print("Computing problem embeddings")
    embeddings = encode(model, graph_data, nodes=args.nodes)

    # Save to path
    res_path = os.path.join(
        "embeddings",
        args.result_infix_path + Path(args.id_file).stem + "_" + args.nodes + ".pkl",
    )
    dump(embeddings, open(res_path, "wb"))
    print("Saved to:", res_path)

    # Make feature matrix
    if args.print_distances:
        print_embedding_distances(embeddings)


if __name__ == "__main__":
    torch.manual_seed(0)
    main()
