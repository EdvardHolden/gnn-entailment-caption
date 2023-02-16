from typing import Dict
import json
import os


def flatten(l):
    return [item for sublist in l for item in sublist]


def load_params(model_dir: str) -> Dict:
    with open(os.path.join(model_dir, "params.json"), "r") as f:
        params = json.load(f)
    return params
