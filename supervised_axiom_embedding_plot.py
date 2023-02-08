"""
Script for computing a PCA plot of the supervised embeddings of the axiom nodes.
"""
import torch
import argparse
from pathlib import Path
from sklearn.decomposition import PCA
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.colors import ListedColormap
import numpy as np

from model import Model
from dataset import get_data_loader
import config
from extract_features import encode, get_activation

parser = argparse.ArgumentParser()
parser.add_argument("--model_path", default="model.pt", help="Path to the model used for embedding")
parser.add_argument("-d", "--dev", default=False, action="store_true", help="Use development set")
parser.add_argument("--plot", choices=["2d", "3d"], default="2d", help="Choose between 2d or 3d PCa plot")
parser.add_argument(
    "--split_label", default=False, action="store_true", help="Plot different plots for true and false"
)

TRAIN_ID = "train.txt"
TRAIN_ID_DEV = "dev_100.txt"

# TEST_ID = "test.txt"

PALETTE = {0: "C0", 1: "C1"}


def get_embedding_data(model, id_file):

    # Get set of problems
    data = get_data_loader(Path(__file__).parent, id_file, batch_size=1, shuffle=False)
    print("Number of problems: ", len(data))

    # Compute model embeddings
    print("Computing problem embeddings")
    embeddings = encode(model, data, nodes="premise", avg=False)

    # Extract embeddings into a single list
    X = []
    for key in sorted(embeddings.keys()):
        X.extend(embeddings[key])

    # First, extract all the values and store in dict
    y_dict = {}
    for n in range(len(data)):
        d = data.dataset.get(n)
        y_dict[d.name] = d.y.detach().numpy()

    # Extract labels into a single list
    y = []
    for key in sorted(embeddings.keys()):
        y.extend(y_dict[key])
    y = [int(label) for label in y]  # Make sure is int

    return X, y


def get_df(data, y):
    column_labels = ["PC" + str(n) for n in range(1, len(data[0]) + 1)]
    df = pd.DataFrame(data, columns=column_labels)
    df["y"] = y
    return df


def main():

    args = parser.parse_args()
    # Load the model
    model = Model(17).to(config.device)
    model.load_state_dict(torch.load(args.model_path))

    # Create hook for getting the intermediate output
    model.dense.register_forward_hook(get_activation("dense"))

    # Get the datasets
    if args.dev:  # Use dev dataset
        X_train, y_train = get_embedding_data(model, TRAIN_ID_DEV)
    else:
        X_train, y_train = get_embedding_data(model, TRAIN_ID)

    # TODO need to clean up this script somehow,
    # Split on true vs false?
    # Might have to set the sizes differently here?

    # Get number of components
    if args.plot == "2d":
        n_components = 2
    else:
        n_components = 3

    # Get the data to plot
    pca = PCA(n_components=n_components)
    train_data = pca.fit_transform(X_train)
    train_df = get_df(train_data, y_train)

    # Compute Limits
    max_lim = np.max(train_data, axis=0)
    min_lim = np.min(train_data, axis=0)

    if args.split_label:
        # Split into true and false labels for better viewing and no occlusions
        plot_dfs = [train_df.loc[train_df["y"] == 1], train_df.loc[train_df["y"] == 0]]
    else:
        plot_dfs = [train_df]

    for df in plot_dfs:
        # Plot
        if args.plot == "2d":
            plot_pca_2d(df, max_lim, min_lim)
        else:
            # FIXME haven't added splitting to 3d as it is hard to get a god viewpoint
            args.plot == "3d"
            plot_pca_3d(df)


def plot_pca_2d(df, max_lim, min_lim):

    # plt.show()
    ax = sns.lmplot(x="PC1", y="PC2", data=df, legend=True, fit_reg=False, hue="y", palette=PALETTE)
    ax.set(xlim=(min_lim[0] - 1, max_lim[0] + 1))
    ax.set(ylim=(min_lim[1] - 1, max_lim[1] + 1))

    plt.show()


def plot_pca_3d(df):

    fig = plt.figure(figsize=(6, 6))
    ax = Axes3D(fig, auto_add_to_figure=False)
    fig.add_axes(ax)
    cmap = ListedColormap(sns.color_palette()[:2])

    # Set axes manually
    sc = ax.scatter(df["PC1"], df["PC2"], df["PC3"], s=40, c=df["y"], marker="o", cmap=cmap, alpha=1)
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_zlabel("PC3")

    # legend
    plt.legend(*sc.legend_elements(), bbox_to_anchor=(1.05, 1), loc=2)
    plt.show()


if __name__ == "__main__":
    torch.manual_seed(0)
    main()
