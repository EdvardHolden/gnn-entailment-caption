#!/usr/bin/env python3
from pathlib import Path
import torch
from torch.nn.functional import binary_cross_entropy_with_logits
from torch.optim import SGD
from torch.optim.lr_scheduler import CyclicLR
from tqdm import tqdm
import argparse

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

    stats = Writer(model)
    best_loss = torch.tensor(float("inf"))

    #  TODO Separate the train and validation loop out?

    # TOdO need better storing of training metrics
    # TODO also include accuracy metrics?

    # TODO change to epochs?
    # while True:
    for _ in range(2):
        stats.report_model_parameters()

        print("validating...")
        model.eval()
        val_loss = validation_loss(model, val_data)
        stats.report_validation_loss(val_loss)
        if val_loss < best_loss:
            torch.save(model.state_dict(), "model_gnn.pt")
            best_loss = val_loss
        print(f"...done, loss {val_loss:.3E}")

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


if __name__ == "__main__":
    torch.manual_seed(0)

    # TODO move to training? - should be removed completely with epochs
    try:
        main()
    except KeyboardInterrupt:
        pass
