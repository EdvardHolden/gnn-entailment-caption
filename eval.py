#!/usr/bin/env python3
import torch
from torch.nn.functional import binary_cross_entropy_with_logits
from torch_geometric.data import Batch
from tqdm import tqdm

from common import mk_loader
from model import Model
from fol_entailment_dataset.dataset import FOLEntailmentDataset

def accuracy(model, data):
    total = 0
    correct = 0
    with torch.no_grad():
        for batch in tqdm(data):
            batch = batch.to('cuda')
            actual = batch.y
            predicted = torch.sigmoid(model(batch)).round().long()
            correct += actual.eq(predicted).sum().item()
            total += len(predicted)
            del batch

    return 100 * (correct / total)

def eval():
    dataset = FOLEntailmentDataset('fol_entailment_dataset/data')
    validation = mk_loader(dataset[98000:99000])
    test = mk_loader(dataset[99000:])

    model = Model(17).to('cuda')
    model.load_state_dict(torch.load('model.pt'))
    model.eval()

    val_acc = accuracy(model, validation)
    test_acc = accuracy(model, test)
    print(f"validation:\t{val_acc:.1f}%")
    print(f"test:\t{test_acc:.1f}%")

if __name__ == '__main__':
    torch.manual_seed(0)
    eval()
