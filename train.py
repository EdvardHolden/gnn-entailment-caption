#!/usr/bin/env python3
from pickle import dump
from pathlib import Path
from typing import Tuple

import torch
from torch.nn.functional import binary_cross_entropy_with_logits
from torch.optim import SGD
from torch.optim.lr_scheduler import CyclicLR
from tqdm import tqdm
import argparse
import numpy as np
from collections import defaultdict
import os

from dataset import get_data_loader, BenchmarkType
from model import Model
from statistics import Writer

import config


def get_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()

    parser.add_argument("--train_id", default="id_files/train.txt", help="ID file used for training")
    parser.add_argument("--val_id", default="id_files/validation.txt", help="ID file used for validation")
    parser.add_argument(
        "--benchmark_type", default="deepmath", type=BenchmarkType, help="Benchmark type fo the problems."
    )
    parser.add_argument(
        "--experiment_dir", default="experiments/premise/test", help="Directory for saving model and stats"
    )
    parser.add_argument("--epochs", default=80, type=int, help="Number of training epochs.")
    parser.add_argument("--es_patience", default=None, type=int, help="Number of EarlyStopping epochs")

    return parser


def batch_loss(model, batch, reduction="mean"):
    batch = batch.to(config.device)
    y = model(batch)
    loss = binary_cross_entropy_with_logits(y, batch.y, reduction=reduction)
    del batch
    return y, loss


def accuracy_matches(pred, actual) -> int:

    # Convert to binary
    predicted = torch.sigmoid(pred).round().long()
    # Compute number of matches
    correct = actual.eq(predicted).sum().item()
    return correct


def dataset_loss(model, dataset) -> Tuple[float, float]:

    losses = []
    tot_correct, tot_predictions = 0, 0

    with torch.no_grad():
        for batch in tqdm(dataset):
            y, loss = batch_loss(model, batch, reduction="none")
            losses.extend(loss.numpy())

            correct = accuracy_matches(y, batch.y)
            tot_correct += correct
            tot_predictions += len(batch.y)

    mean_loss = float(np.mean(losses))
    accuracy = 100 * (tot_correct / tot_predictions)
    return mean_loss, accuracy


def main():

    # Get arguments
    parser = get_parser()
    args = parser.parse_args()

    # Make experiment dir if not exists
    if not os.path.exists(args.experiment_dir):
        os.makedirs(args.experiment_dir)

    train_data = get_data_loader(args.train_id, args.benchmark_type)
    val_data = get_data_loader(args.val_id, args.benchmark_type)

    # Should take model parameters in another way?
    model = Model().to(config.device)
    optimizer = SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=1e-4, nesterov=True)
    scheduler = CyclicLR(optimizer, 0.01, 0.1, mode="exp_range", gamma=0.99995, step_size_up=4000)

    # TODO what is this writer?
    stats = Writer(model)
    best_loss = torch.tensor(float("inf"))

    if args.es_patience is not None:
        print(f"Early Stopping is set to: {args.es_patience}")
        es_wait = 0
        es_best_loss = np.inf

    metrics = defaultdict(list)
    for epoch in range(0, args.epochs):
        print("Epoch", epoch)
        stats.report_model_parameters()

        print("Training...")
        model.train()
        for batch in tqdm(train_data):
            optimizer.zero_grad()
            y, loss = batch_loss(model, batch)
            loss.backward()
            optimizer.step()
            scheduler.step()

            # TODO this is somewhat wrong as the batch size may vary?
            stats.report_train_loss(loss.mean())
            if stats.step % 32 == 0:
                stats.report_output(batch.y, torch.sigmoid(y))
            stats.on_step()

        print("Validating...")
        model.eval()
        train_loss, train_acc = dataset_loss(model, train_data)
        metrics["train_loss"].append(train_loss)
        metrics["train_acc"].append(train_acc)

        val_loss, val_acc = dataset_loss(model, val_data)
        metrics["val_loss"].append(val_loss)
        metrics["val_acc"].append(val_acc)
        stats.report_validation_loss(val_loss)

        # Save the model after every iteration
        torch.save(model.state_dict(), os.path.join(args.experiment_dir, "model_gnn.pt"))

        # Check for early stopping
        if args.es_patience is not None:
            es_wait += 1
            if metrics["val_loss"][-1] < es_best_loss:
                es_best_loss = metrics["val_loss"][-1]
                es_wait = 0
            elif es_wait >= args.es_patience:
                print(f"Terminated training with early stopping after {es_wait} epochs of no improvement")
                break

        # Report epoch metrics
        print(f"Val loss: {val_loss:.3E}")
        print(f"Val acc : {val_acc:.3E}")
        print(f"Train loss: {train_loss:.3E}")
        print(f"Train acc : {train_acc:.3E}")
        print()

    # Save the training history
    print(metrics)
    with open(os.path.join(args.experiment_dir, "history.pkl"), "wb") as f:
        dump(metrics, f)


if __name__ == "__main__":
    torch.manual_seed(0)
    main()
    print("# Finished")
