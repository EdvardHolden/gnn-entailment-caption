#!/usr/bin/env python3

"""Aggregates results from the metrics_eval_best_weights.json in a parent folder"""


import argparse
import os
import pickle

from tabulate import tabulate


parser = argparse.ArgumentParser()
parser.add_argument("result_dir", default="experiments", help="Directory containing results of experiments")
parser.add_argument("--restrict_metric", default=None, nargs="+", help="Restrict table to current metric")


def aggregate_metrics(parent_dir, metrics):
    """Aggregate the metrics of all experiments in folder `parent_dir`.

    Assumes that `parent_dir` contains multiple experiments, with their results stored in
    `parent_dir/subdir/metrics_dev.json`

    Args:
        parent_dir: (string) path to directory containing experiments results
        metrics: (dict) subdir -> {'accuracy': ..., ...}
    """
    # Get the metrics for the folder if it has results from an experiment
    metrics_file = os.path.join(parent_dir, "history.pkl")
    if os.path.isfile(metrics_file):
        with open(metrics_file, "rb") as f:
            data = pickle.load(f)

        # Extract then final evaluation of each step
        scores = {k: v[-1] for k, v in data.items()}
        metrics[parent_dir] = scores

    # Check every subdirectory of parent_dir
    for subdir in os.listdir(parent_dir):
        if not os.path.isdir(os.path.join(parent_dir, subdir)):
            continue
        else:
            aggregate_metrics(os.path.join(parent_dir, subdir), metrics)


def metrics_to_table(metrics):
    # Get the headers from the first subdir. Assumes everything has the same metrics
    headers = metrics[list(metrics.keys())[0]].keys()
    table = [[subdir] + [values[h] for h in headers] for subdir, values in sorted(metrics.items())]
    res = tabulate(table, headers, tablefmt="pipe")

    return res


def main():
    args = parser.parse_args()

    # Aggregate metrics from args.parent_dir directory
    metrics = dict()
    aggregate_metrics(args.result_dir, metrics)

    if args.restrict_metric is not None:
        for exp in metrics:
            metrics[exp] = {k: v for k, v in metrics[exp].items() if k in args.restrict_metric}

    # Create score table
    table = metrics_to_table(metrics)

    # Display the table to terminal
    print(table)

    # Save results in parent_dir/results.md
    save_file = os.path.join(args.result_dir, "results.md")
    with open(save_file, "w") as f:
        f.write(table)


if __name__ == "__main__":
    main()
