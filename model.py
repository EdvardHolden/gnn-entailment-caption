import os
import torch
from torch import nn as nn
from torch.nn import Embedding
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, Linear, MessagePassing, global_mean_pool
from typing import Callable, Optional, Dict

import config
from dataset import LearningTask
from utils import load_params

GCN_NORMALISATION = {"batch": nn.BatchNorm1d, "layer": nn.LayerNorm}


def load_model_params(model_dir: str) -> Dict:

    params = load_params(model_dir)["model"]
    assert params["normalisation"] in GCN_NORMALISATION.keys()
    assert params["gcn_type"] in ["single", "dual_concat", "dual_pool"]
    return params


def get_dense_output_network(task: LearningTask, no_dense_layers: int, hidden_dim: int, dropout_rate: float):
    if task == LearningTask.PREMISE:
        return DenseOutput(hidden_dim, no_dense_layers, hidden_dim, dropout_rate)
    elif task == LearningTask.SIMILARITY:
        return DenseOutput(hidden_dim * 2, no_dense_layers, hidden_dim, dropout_rate)
    else:
        raise ValueError(f"No dense output network for: {task}")


def get_gcn_base(gcn_type: str) -> Callable:
    # TODO this could be a dictionary..

    if gcn_type == "single":
        return GCNDirectional
    elif gcn_type == "dual_concat":
        return GCNDualConcat
    elif gcn_type == "dual_pool":
        return GCNDualPool
    else:
        raise ValueError(f"Unknown gcn type {gcn_type}")


class DenseOutput(torch.nn.Module):
    def __init__(self, input_dim: int, no_dense_layers: int, hidden_dim: int, dropout_rate: float):
        super(DenseOutput, self).__init__()

        self.input_dim = input_dim
        self.no_dense_layers = no_dense_layers
        self.hidden_dim = hidden_dim
        self.dropout_rate = dropout_rate

        self._build_network()

    def _build_network(self):

        if self.no_dense_layers <= 0:  # No hidden layer
            self.out = nn.Linear(self.input_dim, 1)
        else:
            # Add input layer
            self.lin = nn.ModuleList()
            self.lin.append(nn.Linear(self.input_dim, self.hidden_dim))
            # Ad dense layers
            for _ in range(self.no_dense_layers - 1):
                self.lin.append(nn.Linear(self.hidden_dim, self.hidden_dim))
            # Add output layer
            self.out = nn.Linear(self.hidden_dim, 1)

    def forward(self, x):

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

        # TODO refactor this
        # Add normalisation layers used in between graph convolutions
        self.normalisation = normalisation
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


class GCNDualConcat(torch.nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        num_convolutional_layers: int,
        dropout_rate: float,
        normalisation: Optional[str],
        skip_connection: bool,
    ):
        super(GCNDualConcat, self).__init__()

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
        self.normalisation = normalisation

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


class GCNDualPool(torch.nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        num_convolutional_layers: int,
        dropout_rate: float,
        normalisation: Optional[str],
        skip_connection: bool,
    ):
        super(GCNDualPool, self).__init__()

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
        self.normalisation = normalisation

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

            # Compute the average embedding
            x = (x_up + x_down) / 2
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout_rate, training=self.training)

            # Normalise, if set
            if self.lns is not None and not i == self.num_convolutional_layers - 1:  # Apply normalisation
                x = self.lns[i](x)

        return emb, x


class GNNStack(torch.nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        num_convolutional_layers: int,
        no_dense_layers: int,
        gcn_type: str,
        dropout_rate: float = 0.0,
        no_embeddings: int = len(config.NODE_TYPE),
        task: LearningTask = LearningTask.PREMISE,
        normalisation="layer",
        skip_connection: bool = True,
    ):
        super(GNNStack, self).__init__()

        # Set variables
        self.task = task
        self.dropout_rate = dropout_rate
        self.hidden_dim = hidden_dim

        # Add embedding layer
        self.node_embedding = Embedding(no_embeddings, hidden_dim)

        # Add GCN layer(s)
        self.gcn_type = gcn_type
        gcn_base = get_gcn_base(self.gcn_type)

        self.gcn = gcn_base(
            hidden_dim=self.hidden_dim,
            num_convolutional_layers=num_convolutional_layers,
            dropout_rate=self.dropout_rate,
            normalisation=normalisation,
            skip_connection=skip_connection,
        )

        # Post-message-passing
        self.post_mp = get_dense_output_network(
            self.task, no_dense_layers, hidden_dim, dropout_rate=self.dropout_rate
        )

    def forward(self, data):
        x, edge_index, premise_index = data.x, data.edge_index, data.premise_index

        x = self.node_embedding(x)

        emb, x = self.gcn(x, edge_index)

        # Call feedforward layer on the premise indexes
        x = self.post_mp(x[premise_index])

        return emb, x

    """
    def __str__(self) -> str:
        # tODO should really be nested???
        return (
            f"GNNStack:hidden_dim_{self.hidden_dim}_num_convolution_layers_{self.gcn.num_convolutional_layers}"
            f"_no_dense_layers{self.post_mp.no_dense_layers}_direction_{self.direction}_drop_out_rate_{self.dropout_rate}"
            f"_normalisation_{self.gcn.normalisation}_skip_connection_{self.gcn.skip_connection}"
        )
    """


class GNNStackSiamese(GNNStack):
    def __init__(self, **kwargs):
        kwargs["task"] = LearningTask.SIMILARITY
        super(GNNStackSiamese, self).__init__(**kwargs)

    def forward(self, data):
        # Extract data
        x_s, edge_index_s = data.x_s, data.edge_index_s
        x_t, edge_index_t = data.x_t, data.edge_index_t

        # Embed node types
        x_s = self.node_embedding(x_s)
        x_t = self.node_embedding(x_t)

        # Embed s and t graphs using the same neural network
        emb_s, x_s = self.gcn(x_s, edge_index_s)
        emb_t, x_t = self.gcn(x_t, edge_index_t)

        # Pool the graph
        x_s = global_mean_pool(x_s, data.x_s_batch)
        x_t = global_mean_pool(x_t, data.x_t_batch)

        # Concat embeddings
        x = torch.cat((x_s, x_t), dim=1)

        x = self.post_mp(x)

        return (emb_s, emb_t), x


def load_model(model_dir, learning_task):
    model = get_model(model_dir, learning_task)
    model.load_state_dict(torch.load(os.path.join(model_dir, "model_gnn.pt")))

    return model


def get_model(experiment_dir: str, learning_task: LearningTask) -> nn.Module:
    model_params = load_model_params(experiment_dir)
    model_params["task"] = learning_task  # Set task from input
    if learning_task == LearningTask.PREMISE:
        model = GNNStack(**model_params)
    elif learning_task == LearningTask.SIMILARITY:
        model = GNNStackSiamese(**model_params)
    else:
        raise ValueError()

    model = model.to(config.device)
    return model


def main():

    test_model = GNNStack(
        hidden_dim=24,
        num_convolutional_layers=8,
        no_dense_layers=1,
        direction="separate",
        dropout_rate=0.25,
    )
    from torchinfo import summary

    summary(test_model)


if __name__ == "__main__":
    main()
