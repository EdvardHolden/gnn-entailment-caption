#!/usr/bin/env python3
from pathlib import Path
import torch
from tqdm import tqdm

from common import mk_loader
from model import Model

import config
from utils import read_problem_deepmath

DEST = "generated_problems_merged_retry/"


def predict_labels(model, batch):
    return torch.sigmoid(model(batch)).round().long()


def select_premises(model, data):
    with torch.no_grad():
        for batch in tqdm(data):
            batch = batch.to(config.device)
            assert len(batch.name) == 1
            prob_name = batch.name[0]

            # Predict
            predictions = predict_labels(model, batch)

            # Get hold of original problem
            conj, premises, _ = read_problem_deepmath(prob_name, ".")

            # Set conjecture properly
            conj = [conj[0].replace(b"axiom", b"conjecture", 1)]

            # Filter based on labels
            premises = [prem.strip() for prem, pred in zip(premises, predictions) if pred]

            # Set problem from conjecture and selected premises
            problem = conj + premises

            # Write resulting problem to file
            with open(DEST + prob_name, "wb") as f:
                f.write(b"\n".join(problem) + b"\n")

            del batch


def eval():

    # Load dataset
    # data_ids = "single.txt"
    #data_ids = "deepmath.txt"
    data_ids = "deepmath_merged.txt"
    data = mk_loader(Path(__file__).parent, data_ids, batch_size=1)

    # Laod and prepare model
    model = Model(17).to(config.device)
    model.load_state_dict(torch.load("model_gnn.pt"))
    model.eval()

    # Run premise selection
    select_premises(model, data)

    print("## FINISHED")


if __name__ == "__main__":
    torch.manual_seed(0)
    eval()
