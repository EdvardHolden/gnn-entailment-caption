from model import load_model, get_model
from utils import flatten
from dataset import LearningTask


def get_transfer_model(model_path: str, new_task: LearningTask):

    # Determine the old task - only two available
    if new_task is LearningTask.PREMISE:
        model_task = LearningTask.SIMILARITY
    elif new_task is LearningTask.SIMILARITY:
        model_task = LearningTask.PREMISE
    else:
        raise ValueError(f"Could not determine original task  from '{new_task}' in get_transfer_model")

    # Load trained model
    gnn_model = load_model(model_path, model_task)

    # Get same model as previous - but for the new task
    new_model = get_model(model_path, new_task)

    # Set the embedding and gcn layers
    new_model.node_embedding = gnn_model.node_embedding
    new_model.gcn = gnn_model.gcn

    # Make these components non-trainable - only want to train the dense output layer
    for param in gnn_model.node_embedding.parameters():
        param.requires_grad = False
    for param in gnn_model.gcn.parameters():
        param.requires_grad = False

    return new_model


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
