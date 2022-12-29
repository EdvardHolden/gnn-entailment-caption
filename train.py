#!/usr/bin/env python3
from pathlib import Path
import torch
from torch.nn.functional import binary_cross_entropy_with_logits
from torch.optim import SGD
from torch.optim.lr_scheduler import CyclicLR
from tqdm import tqdm
import argparse
import numpy as np
from collections import defaultdict

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
    parser.add_argument("--epochs", default=80, type=int, help="Number of training epochs.")
    parser.add_argument("--es_patience", default=None, type=int, help="Number of EarlyStopping epochs")

    return parser


def batch_loss(model, batch):
    batch = batch.to(config.device)
    y = model(batch)
    loss = binary_cross_entropy_with_logits(y, batch.y)
    del batch
    return y, loss


def validation_loss(model, validation):
    losses = []
    with torch.no_grad():
        for batch in tqdm(validation):
            _, loss = batch_loss(model, batch)
            losses.append(loss.mean())

    return torch.tensor(losses).mean()


def main():

    # Get arguments
    parser = get_parser()
    args = parser.parse_args()

    train_data = get_data_loader(args.train_id, args.benchmark_type)
    val_data = get_data_loader(args.val_id, args.benchmark_type)

    # Should take model parameters in another way?
    model = Model().to(config.device)
    optimizer = SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=1e-4, nesterov=True)
    scheduler = CyclicLR(optimizer, 0.01, 0.1, mode="exp_range", gamma=0.99995, step_size_up=4000)

    # TODO what is this writer?
    stats = Writer(model)
    best_loss = torch.tensor(float("inf"))

    #  TODO Separate the train and validation loop out?

    # TOdO need better storing of training metrics
    # TODO also include accuracy metrics?

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

            stats.report_train_loss(loss.mean())
            if stats.step % 32 == 0:
                stats.report_output(batch.y, torch.sigmoid(y))
            stats.on_step()

        print("Validating...")
        model.eval()
        val_loss = validation_loss(model, val_data)
        metrics["val_loss"].append(float(val_loss.numpy()))
        stats.report_validation_loss(val_loss)

        # Save the model after every iteration
        torch.save(model.state_dict(), "model_gnn.pt")

        # Check for early stopping
        if args.es_patience is not None:
            es_wait += 1
            if metrics["val_loss"][-1] < es_best_loss:
                es_best_loss = metrics["val_loss"][-1]
                es_wait = 0
            elif es_wait >= args.es_patience:
                print(f"Terminated training with early stopping after {es_wait} epochs of no improvement")
                break

        # TODO report both losses
        print(f"Val loss: {val_loss:.3E}")
        print()

    # TODO save some overall stats?
    print(metrics)


if __name__ == "__main__":
    torch.manual_seed(0)
    main()
    print("# Finished")
