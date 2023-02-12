import torch
import os

from train import get_model
from utils import flatten


def load_model(model_dir, learning_task):
    model = get_model(model_dir, learning_task)
    model.load_state_dict(torch.load(os.path.join(model_dir, "model_gnn.pt")))

    return model


def get_model_embedding(model, batch):
    model.eval()
    node_emb = model.node_embedding(batch.x)
    emb, _ = model.gcn(node_emb, batch.edge_index)

    return emb


def get_node_embeddings(model, batch):
    emb = get_model_embedding(model, batch)

    group = flatten([batch.x.numpy()])

    return emb, group


def get_premise_embeddings(model, batch):
    emb = get_model_embedding(model, batch)

    emb = emb[batch.premise_index]
    group = flatten([batch.y.numpy()])

    return emb, group
