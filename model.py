import os
import json
import torch
from torch.nn import Embedding
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, Linear, MessagePassing
import torch.nn as nn
from typing import Callable, Optional, Dict

import config

GCN_NORMALISATION = {"batch": nn.BatchNorm1d, "layer": nn.LayerNorm}


def load_model_params(model_dir: str) -> Dict:
    # Load parameters from model directory and create namespace
    with open(os.path.join(model_dir, "params.json"), "r") as f:
        params = json.load(f)
    return params


def get_dense_output_network(
    no_dense_layers: int, hidden_dim: int, task: str, dropout_rate: float
) -> nn.Module:
    if task == "premise":
        return DensePremiseOutput(no_dense_layers, hidden_dim, dropout_rate)
    else:
        raise NotImplementedError


class DensePremiseOutput(torch.nn.Module):
    def __init__(self, no_dense_layers, hidden_dim, dropout_rate):
        super(DensePremiseOutput, self).__init__()

        self.no_dense_layers = no_dense_layers
        self.hidden_dim = hidden_dim
        self.dropout_rate = dropout_rate

        self.lin = nn.ModuleList()
        for _ in range(no_dense_layers):
            self.lin.append(nn.Linear(hidden_dim, hidden_dim))

        # Add output layer
        self.out = nn.Linear(hidden_dim, 1)

    def forward(self, x, premise_index):

        # Extract premises
        x = x[premise_index]

        # Dense feedforward
        for i in range(self.no_dense_layers):
            x = self.lin[i](x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout_rate, training=self.training)

        # Output layer
        x = self.out(x)
        x = x.squeeze(-1)

        return x


def build_normalisation_layers(
    normaliser: Callable, hidden_dim: int, num_normalisation_layers: int
) -> nn.ModuleList:

    lns = nn.ModuleList()
    for _ in range(num_normalisation_layers):
        lns.append(normaliser(hidden_dim))
    return lns


def build_conv_model(num_convolutional_layers: int, hidden_dim: int, flow: str) -> nn.ModuleList:

    convs = nn.ModuleList()
    for _ in range(num_convolutional_layers):
        convs.append(get_conv_layer(hidden_dim, hidden_dim, flow))

    return convs


def get_conv_layer(input_dim: int, hidden_dim: int, flow: str) -> MessagePassing:
    return GCNConv(input_dim, hidden_dim, flow=flow)


def build_merge_linear_layers(num_linear_layers: int, hidden_dim: int) -> nn.ModuleList:

    lin = nn.ModuleList()
    for _ in range(num_linear_layers):
        lin.append(Linear(hidden_dim * 2, hidden_dim))

    return lin


class GCNDirectional(torch.nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        num_convolutional_layers: int,
        dropout_rate: float,
        normalisation: Optional[str],
        skip_connection: bool,
    ):
        super(GCNDirectional, self).__init__()

        self.flow = "target_to_source"  # Sets direction to bottom up

        # Set variables
        self.hidden_dim = hidden_dim
        self.num_convolutional_layers = num_convolutional_layers
        self.dropout_rate = dropout_rate
        self.skip_connection = skip_connection

        # Add convolutional layers
        self.convs = build_conv_model(num_convolutional_layers, hidden_dim, self.flow)

        # Add normalisation layers used in between graph convolutions
        if normalisation is None:
            self.lns = None
        else:
            self.normaliser = GCN_NORMALISATION[normalisation]
            self.lns = build_normalisation_layers(self.normaliser, hidden_dim, num_convolutional_layers - 1)

    def forward(self, x, edge_index):

        # Iterate over each convolutional sequence
        emb = None
        for i in range(self.num_convolutional_layers):

            conv_out = self.convs[i](x, edge_index)
            # Check if applying skip connection
            if self.skip_connection:
                x = x + conv_out
            else:
                x = conv_out

            emb = x
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout_rate, training=self.training)

            if self.lns is not None and not i == self.num_convolutional_layers - 1:  # Apply normalisation
                x = self.lns[i](x)

        return emb, x


class GCNBiDirectional(torch.nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        num_convolutional_layers: int,
        dropout_rate: float,
        normalisation: Optional[str],
        skip_connection: bool,
    ):
        super(GCNBiDirectional, self).__init__()

        # Set variables
        self.hidden_dim = hidden_dim
        self.num_convolutional_layers = num_convolutional_layers
        self.dropout_rate = dropout_rate
        self.skip_connection = skip_connection

        # Add convolutional layers
        self.convs_up = build_conv_model(num_convolutional_layers, hidden_dim, flow="source_to_target")
        self.convs_down = build_conv_model(num_convolutional_layers, hidden_dim, flow="target_to_source")

        # Add normalisation layers used in between graph convolutions
        if normalisation is None:
            self.lns = None
        else:
            self.normaliser = GCN_NORMALISATION[normalisation]
            self.lns = build_normalisation_layers(self.normaliser, hidden_dim, num_convolutional_layers - 1)

        # Add Linear layers
        self.linear = build_merge_linear_layers(num_convolutional_layers, hidden_dim)

    def forward(self, x, edge_index):

        # Iterate over each convolutional sequence
        emb = None
        for i in range(self.num_convolutional_layers):

            # Apply convolutions
            x_up = self.convs_up[i](x, edge_index)
            x_down = self.convs_down[i](x, edge_index)

            # Check if applying skip connection
            if self.skip_connection:
                x_up = x + x_up
                x_down = x + x_down

            # Concat convolutions
            x = torch.cat((x_up, x_down), dim=1)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout_rate, training=self.training)

            # Merge through linear
            x = self.linear[i](x)
            emb = x
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout_rate, training=self.training)

            # Normalise, if set
            if self.lns is not None and not i == self.num_convolutional_layers - 1:  # Apply normalisation
                x = self.lns[i](x)

        return emb, x


class GNNStack(torch.nn.Module):
    def __init__(
        self,
        hidden_dim,
        num_convolutional_layers,
        no_dense_layers,
        direction,
        dropout_rate=0.0,
        task="premise",
        normalisation="layer",
        skip_connection=True,
    ):
        super(GNNStack, self).__init__()

        # Set variables
        self.task = task
        self.dropout_rate = dropout_rate
        self.hidden_dim = hidden_dim

        # Add embedding layer
        self.node_embedding = Embedding(len(config.NODE_TYPE), hidden_dim)

        # Add GCN layer
        if direction == "single":
            gcn_base = GCNDirectional
        elif direction == "separate":
            gcn_base = GCNBiDirectional
        else:
            raise ValueError(f"Unknown gcn direction {direction}")

        self.gcn = gcn_base(
            hidden_dim=self.hidden_dim,
            num_convolutional_layers=num_convolutional_layers,
            dropout_rate=self.dropout_rate,
            normalisation=normalisation,
            skip_connection=skip_connection,
        )

        # Post-message-passing
        self.post_mp = get_dense_output_network(
            no_dense_layers, hidden_dim, task=self.task, dropout_rate=self.dropout_rate
        )

    def forward(self, data):
        x, edge_index, premise_index = data.x, data.edge_index, data.premise_index

        x = self.node_embedding(x)

        emb, x = self.gcn(x, edge_index)

        if self.task == "premise":
            x = self.post_mp(x, premise_index)
        else:
            raise NotImplementedError()

        return emb, x


if __name__ == "__main__":

    test_model = GNNStack(
        hidden_dim=32,
        num_convolutional_layers=3,
        no_dense_layers=1,
        direction="single",
        dropout_rate=0.25,
        task="premise",
    )
    print(test_model)
