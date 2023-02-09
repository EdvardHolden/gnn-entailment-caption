#!/usr/bin/env python3

import torch
import argparse
import numpy as np
import os
from torch_geometric.transforms import ToUndirected
import torch.nn.functional as F
from typing import Tuple

import config
from dataset import get_data_loader, BenchmarkType, LearningTask
from model import GNNStack, load_model_params
from stats_writer import Writer


def get_train_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()

    parser.add_argument("--train_id", default=config.TRAIN_ID, help="ID file used for training")
    parser.add_argument("--val_id", default=config.VAL_ID, help="ID file used for validation")
    parser.add_argument(
        "--benchmark_type",
        default="deepmath",
        choices=BenchmarkType.list(),
        type=BenchmarkType,
        help="Benchmark type fo the problems.",
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
        default="premise",
        choices=LearningTask.list(),
        type=LearningTask,
        help="Learning task for training the GCN model",
    )

    parser.add_argument("--graph_bidirectional", action="store_true", help="Makes the graphs bidirectional")
    parser.add_argument(
        "--graph_remove_argument_node", action="store_true", help="Removes the argument nodes from the graphs"
    )

    return parser


def train_step(model, train_data, criterion, optimizer):
    model.train()

    for data in train_data:  # Iterate in batches over the training dataset.
        data = data.to(config.device)
        _, out = model(data)  # Perform a single forward pass.

        loss = criterion(out, data.y)  # Compute the loss.
        loss.backward()  # Derive gradients.
        optimizer.step()  # Update parameters based on gradients.
        optimizer.zero_grad()  # Clear gradients.


def test_step(model, test_data, writer: Writer, testing: bool) -> Tuple[float, float]:
    model.eval()

    correct = 0
    total_samples = 0
    total_loss = 0.0

    for data in test_data:  # Iterate in batches over the training/test dataset.
        data = data.to(config.device)
        _, out = model(data)

        # Compute accuracy
        pred = torch.sigmoid(out).round().long()
        correct += data.y.eq(pred).sum().item()

        # Compute loss
        loss = F.binary_cross_entropy_with_logits(out, data.y, reduction="sum")
        total_loss += loss
        total_samples += len(pred)

    acc_score = correct / total_samples  # Derive ratio of correct predictions.
    total_loss /= total_samples
    total_loss = total_loss.item()

    if testing:
        writer.report_val_score(acc_score)
        writer.report_val_loss(total_loss)
    else:
        writer.report_train_score(acc_score)
        writer.report_train_loss(total_loss)

    return total_loss, acc_score


def main():
    # Get arguments
    parser = get_train_parser()
    args = parser.parse_args()

    dataset_params = {}
    if args.graph_bidirectional:
        dataset_params["transform"] = ToUndirected()
    if args.graph_remove_argument_node:
        dataset_params["remove_argument_node"] = True

    # Make experiment dir if not exists
    if not os.path.exists(args.experiment_dir):
        os.makedirs(args.experiment_dir)

    train_data = get_data_loader(args.train_id, args.benchmark_type, **dataset_params)
    val_data = get_data_loader(args.val_id, args.benchmark_type, **dataset_params)

    # Initialise model
    model_params = load_model_params(args.experiment_dir)
    model_params["task"] = args.learning_task  # Set task from input
    model = GNNStack(**model_params)
    model = model.to(config.device)

    # Initialise writer
    writer = Writer(model)

    # Training optimisers
    # optimizer = SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=1e-4, nesterov=True)
    # scheduler = CyclicLR(optimizer, 0.01, 0.1, mode="exp_range", gamma=0.99995, step_size_up=4000)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)

    # TODO change criterion thing here!
    criterion = torch.nn.BCEWithLogitsLoss(reduction="mean")

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

        train_loss, train_acc = test_step(model, train_data, writer, testing=False)
        test_loss, test_acc = test_step(model, val_data, writer, testing=True)
        print(
            f"Epoch: {epoch:03d}, Train Loss: {train_loss:.4f}, Test Loss: {test_loss:.4f}, "
            f"Train Acc: {train_acc:.4f}, Test Acc: {test_acc:.4f}"
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

    # Save the training history
    writer.save_scores(args.experiment_dir)


if __name__ == "__main__":
    torch.manual_seed(1234)
    main()
    print("# Finished")
