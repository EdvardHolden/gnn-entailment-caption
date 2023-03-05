#!/usr/bin/env python3
import argparse
from pathlib import Path
import torch
from tqdm import tqdm

import config
from model import load_model
from train import get_score
from dataset import BenchmarkType, LearningTask, get_data_loader, load_graph_params


def get_evaluate_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()

    parser.add_argument("model_dir", help="Path to the model to evaluate")
    parser.add_argument(
        "learning_task",
        # default=LearningTask.PREMISE,
        choices=list(LearningTask),
        type=lambda x: LearningTask(x),
        help="Learning task of the model (Must match the original training objective)",
    )

    parser.add_argument("--id_file", default=config.TEST_ID, help="ID file used for evaluation")
    parser.add_argument(
        "--in_memory", action="store_true", help="Set dataset to in memory (may not always work)"
    )

    parser.add_argument(
        "--benchmark_type",
        default=BenchmarkType.DEEPMATH,
        choices=list(BenchmarkType),
        type=lambda x: BenchmarkType(x),
        help="Benchmark type of the problems.",
    )
    return parser


def score_model(model, data, task):
    score = 0
    total_samples = 0
    with torch.no_grad():
        for batch in tqdm(data):
            batch = batch.to(config.device)
            _, out = model(batch)

            # Compute score
            score += get_score(task, out, batch.y)
            total_samples += len(out)

    total_score = score / total_samples  # Derive average score
    return total_score


def evaluate() -> None:

    # Get the parser
    parser = get_evaluate_parser()
    args = parser.parse_args()

    # Load the model
    model = load_model(args.model_dir, args.learning_task)
    model = model.to(config.device)
    model.eval()

    # Get data loader
    graph_params = load_graph_params(args.model_dir)
    eval_data = get_data_loader(
        args.id_file, args.benchmark_type, task=args.learning_task, in_memory=args.in_memory, **graph_params
    )

    eval_score = score_model(model, eval_data, args.learning_task)
    print(f"Evaluation score: {eval_score:.2f}")


if __name__ == "__main__":
    torch.manual_seed(0)
    evaluate()
