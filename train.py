#!/usr/bin/env python3

import torch
import argparse
import numpy as np
import os
from torch_geometric.transforms import ToUndirected
import torch_geometric as pyg
import torch_geometric.nn as pyg_nn
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Callable

import config
from dataset import get_data_loader, BenchmarkType, LearningTask
from model import GNNStackSiamese, GNNStack, load_model_params
from stats_writer import Writer


def get_train_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()

    parser.add_argument("--train_id", default=config.TRAIN_ID, help="ID file used for training")
    parser.add_argument("--val_id", default=config.VAL_ID, help="ID file used for validation")
    parser.add_argument(
        "--benchmark_type",
        default=BenchmarkType.DEEPMATH,
        choices=list(BenchmarkType),
        type=lambda x: BenchmarkType(x),
        help="Benchmark type of the problems.",
    )
    parser.add_argument(
        "--experiment_dir", default="experiments/premise/test", help="Directory for saving model and stats"
    )
    parser.add_argument("--epochs", default=config.EPOCHS, type=int, help="Number of training epochs.")
    parser.add_argument(
        "--es_patience", default=config.ES_PATIENCE, type=int, help="Number of EarlyStopping epochs"
    )

    parser.add_argument(
        "--learning_task",
        default=LearningTask.PREMISE,
        choices=list(LearningTask),
        type=lambda x: LearningTask(x),
        help="Learning task for training the GCN model",
    )

    parser.add_argument("--graph_bidirectional", action="store_true", help="Makes the graphs bidirectional")
    parser.add_argument(
        "--graph_remove_argument_node", action="store_true", help="Removes the argument nodes from the graphs"
    )

    return parser


def train_step(model: nn.Module, train_data: pyg.loader.DataLoader, criterion, optimizer) -> None:
    model.train()

    for data in train_data:  # Iterate in batches over the training dataset.
        data = data.to(config.device)
        _, out = model(data)  # Perform a single forward pass.

        loss = criterion(out, data.y, reduction="mean")  # Compute the loss.
        loss.backward()  # Derive gradients.
        optimizer.step()  # Update parameters based on gradients.
        optimizer.zero_grad()  # Clear gradients.


def get_score(task, out, y):

    if task == LearningTask.PREMISE:
        # Premise used accuracy
        pred = torch.sigmoid(out).round().long()
        score = y.eq(pred).sum().item()
    elif task == LearningTask.SIMILARITY:
        score = F.l1_loss(out, y, reduction="sum")
    else:
        raise ValueError()

    return score


def test_step(
    model: nn.Module,
    test_data: pyg.loader.DataLoader,
    writer: Writer,
    criterion,
    task: LearningTask,
    testing: bool,
) -> Tuple[float, float]:
    model.eval()

    score = 0
    total_samples = 0
    total_loss = 0.0

    for data in test_data:  # Iterate in batches over the training/test dataset.
        data = data.to(config.device)
        _, out = model(data)

        # Compute score
        score += get_score(task, out, data.y)

        # Compute loss
        loss = criterion(out, data.y, reduction="sum")
        total_loss += loss
        total_samples += len(out)

    score = score / total_samples  # Derive average score
    total_loss /= total_samples
    total_loss = total_loss.item()

    if testing:
        writer.report_val_score(score)
        writer.report_val_loss(total_loss)
    else:
        writer.report_train_score(score)
        writer.report_train_loss(total_loss)

    return total_loss, score


def get_criterion(learning_task: LearningTask) -> Callable:
    # Select criterion loss based on learning task
    if learning_task == LearningTask.PREMISE:
        criterion = F.binary_cross_entropy_with_logits
    elif learning_task == LearningTask.SIMILARITY:
        criterion = F.mse_loss
    else:
        raise ValueError(f"Cannot select cooperation based on task {learning_task}")

    return criterion


def get_model(experiment_dir: str, learning_task: LearningTask) -> nn.Module:
    model_params = load_model_params(experiment_dir)
    model_params["task"] = learning_task  # Set task from input
    if learning_task == LearningTask.PREMISE:
        model = GNNStack(**model_params)
    elif learning_task == LearningTask.SIMILARITY:
        model = GNNStackSiamese(**model_params)
    else:
        raise ValueError()

    model = model.to(config.device)
    return model


def main():
    # Get arguments
    parser = get_train_parser()
    args = parser.parse_args()
    learning_task = args.learning_task

    dataset_params = {}
    if args.graph_bidirectional:
        dataset_params["transform"] = ToUndirected()
    if args.graph_remove_argument_node:
        dataset_params["remove_argument_node"] = True

    # Make experiment dir if not exists
    if not os.path.exists(args.experiment_dir):
        os.makedirs(args.experiment_dir)

    # Get datasets wrapper in a loader
    train_data = get_data_loader(args.train_id, args.benchmark_type, task=learning_task, **dataset_params)
    val_data = get_data_loader(args.val_id, args.benchmark_type, task=learning_task, **dataset_params)

    # Initialise model
    model = get_model(args.experiment_dir, learning_task)
    # Initialise writer
    writer = Writer(model)

    # Training optimisers
    # optimizer = SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=1e-4, nesterov=True)
    # scheduler = CyclicLR(optimizer, 0.01, 0.1, mode="exp_range", gamma=0.99995, step_size_up=4000)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
    criterion = get_criterion(learning_task)

    # Set up training
    if args.es_patience is not None:
        print(f"Early Stopping is set to: {args.es_patience}")
        es_wait = 0
        es_best_loss = np.inf
    else:
        es_wait, es_best_loss = None, None

    for epoch in range(0, args.epochs):

        train_step(model, train_data, criterion, optimizer)
        # writer.report_model_parameters() # FIXME - crashes for unknown reason...

        train_loss, train_score = test_step(
            model, train_data, writer, criterion, learning_task, testing=False
        )
        test_loss, test_score = test_step(model, val_data, writer, criterion, learning_task, testing=True)
        print(
            f"Epoch: {epoch:03d}, Train Loss: {train_loss:.4f}, Test Loss: {test_loss:.4f}, "
            f"Train Score: {train_score:.4f}, Test Score: {test_score:.4f}"
        )

        # Check for early stopping
        if args.es_patience is not None:
            es_wait += 1
            if test_loss < es_best_loss:
                es_best_loss = test_loss
                es_wait = 0
                # Always save the best model
                torch.save(model.state_dict(), os.path.join(args.experiment_dir, "model_gnn.pt"))
            elif es_wait >= args.es_patience:
                print(f"Terminated training with early stopping after {es_wait} epochs of no improvement")
                break
        else:
            # Always save model if no ES
            torch.save(model.state_dict(), os.path.join(args.experiment_dir, "model_gnn.pt"))

        # Increment epoch for next iteration
        writer.on_step()

    # TODO check on test data?

    # Save the training history
    writer.save_scores(args.experiment_dir)


if __name__ == "__main__":
    torch.manual_seed(1234)
    main()
    print("# Finished")
