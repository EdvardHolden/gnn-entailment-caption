#!/usr/bin/env python3
from pathlib import Path
import torch
from torch.nn.functional import binary_cross_entropy_with_logits
from torch_geometric.data import Batch
from tqdm import tqdm

from dataset import get_data_loader
from model import Model

import config


def accuracy(model, data):
    total = 0
    correct = 0
    with torch.no_grad():
        for batch in tqdm(data):
            batch = batch.to(config.device)
            actual = batch.y
            predicted = torch.sigmoid(model(batch)).round().long()
            correct += actual.eq(predicted).sum().item()
            total += len(predicted)
            del batch

    return 100 * (correct / total)


def eval():
    test = get_data_loader(Path(__file__).parent, "test.txt")
    model = Model(17).to(config.device)
    # model.load_state_dict(torch.load('model.pt'))
    model.load_state_dict(torch.load("model_gnn.pt"))
    model.eval()

    test_acc = accuracy(model, test)
    print(f"test:\t{test_acc:.1f}%")


if __name__ == "__main__":
    torch.manual_seed(0)
    eval()
