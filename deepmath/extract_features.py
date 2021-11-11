import torch
from tqdm import tqdm
from pickle import dump
from pathlib import Path

from common import mk_loader
from model import Model
import config

# Create hooks
activation = {}


def get_activation(name):
    def hook(model, input, output):
        activation[name] = output.detach()
    return hook


def encode(model, data):
    embeddings = {}
    with torch.no_grad():
        for batch in tqdm(data):
            batch = batch.to(config.device)
            _ = model(batch)

            # Get output of dense layer
            emb = activation['dense']

            # Get the mean of the graph (might change to node/conj indecies?)
            e = torch.mean(emb, 0)
            e = e.detach().numpy()

            # Batch size is 1 for this setup
            embeddings[batch.name[0]] = e
            del batch

    return embeddings


def main():

    # Get set of problems
    #data = mk_loader(Path(__file__).parent, 'validation.txt', batch_size=1, shuffle=False)
    data = mk_loader(Path(__file__).parent, 'example.txt', batch_size=1, shuffle=False)

    # Load the model
    model = Model(17).to(config.device)
    model.load_state_dict(torch.load('model.pt'))

    # Create hook for getting the intermediate output
    model.dense.register_forward_hook(get_activation('dense'))

    # Compute model embeddings
    print("Computing problem embeddings")
    embeddings = encode(model, data)
    #print(embeddings)

    # Save to path
    dump(embeddings, open('graph_features.pkl', 'wb'))

    from scipy.spatial.distance import euclidean, cosine
    from sklearn.metrics import pairwise_distances

    # Make feature matrix
    feat = []
    for key in sorted(embeddings.keys()):
        feat.append(embeddings[key])

    import numpy as np
    np.set_printoptions(precision=3)
    np.set_printoptions(suppress=True)

    print("# Euclidean distances")
    print(pairwise_distances(feat, metric='euclidean'))
    print()
    print("# Cosine distances ")
    print(pairwise_distances(feat, metric='cosine'))
    print()

if __name__ == "__main__":
    # For the proof of concept, we only extract features from train.txt
    torch.manual_seed(0)
    main()
