import torch
from torch.nn import BatchNorm1d, Dropout, Linear, Module, ModuleList
from torch.nn.functional import relu
from torch_geometric import nn as geom
from torch_geometric.utils import to_undirected, dropout_adj

EDGE_DROPOUT = 0#0.005
FC_DROPOUT = 0.1
K = 16

class FullyConnectedLayer(Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.bn = BatchNorm1d(in_channels)
        self.dropout = Dropout(p=FC_DROPOUT)
        self.linear = Linear(in_channels, out_channels)

    def forward(self, x):
        x = self.bn(x)
        x = self.dropout(x)
        x = self.linear(x)
        x = relu(x)
        return x

class EdgeDropoutLayer(Module):
    def __init__(self):
        super().__init__()

    def forward(self, edge_index):
        edge_index, _ = dropout_adj(
            edge_index,
            p=EDGE_DROPOUT,
            training=self.training
        )
        return edge_index

class EmbeddingLayer(Module):
    def __init__(self):
        super().__init__()
        self.embed = Linear(4, K)

    def forward(self, x):
        x = self.embed(x)
        return x

class InputBlock(Module):
    def __init__(self):
        super().__init__()
        self.edge_dropout = EdgeDropoutLayer()
        self.embed = EmbeddingLayer()

    def forward(self, x, edge_index):
        edge_index = self.edge_dropout(edge_index)
        x = self.embed(x)
        return (x, edge_index)

class ConvLayer(Module):
    def __init__(self, in_channels):
        super().__init__()
        self.bn = BatchNorm1d(in_channels)
        self.conv_in = geom.GCNConv(in_channels, K // 2)
        self.conv_out = geom.GCNConv(
            in_channels,
            K // 2,
            flow='target_to_source'
        )

    def forward(self, x, edge_index):
        x = self.bn(x)
        x = relu(x)
        in_x = self.conv_in(x, edge_index)
        out_x = self.conv_out(x, edge_index)
        x = torch.cat((in_x, out_x), dim=1)
        return x

class DenseBlock(Module):
    def __init__(self, layers):
        super().__init__()
        self.fc = ModuleList([
            FullyConnectedLayer(K * (layer + 1), 2 * K)
            for layer in range(layers)
        ])
        self.conv = ModuleList([
            ConvLayer(2 * K)
            for _ in range(layers)
        ])

    def forward(self, x, edge_index):
        outputs = [x]
        for conv, fc in zip(self.conv, self.fc):
            combined = torch.cat(outputs, dim=1)
            x = fc(combined)
            x = conv(x, edge_index)
            outputs.append(x)

        x = torch.cat(outputs[1:], dim=1)
        return x

class GlobalPoolLayer(Module):
    def __init__(self, in_channels):
        super().__init__()
        self.bn = BatchNorm1d(in_channels)

    def forward(self, x, batch):
        x = self.bn(x)
        max_pooled = geom.global_max_pool(x, batch)
        mean_pooled = geom.global_mean_pool(x, batch)
        x = torch.cat((max_pooled, mean_pooled), dim=1)
        return x

class OutputBlock(Module):
    def __init__(self, in_channels, hidden_channels):
        super().__init__()
        self.fc1 = FullyConnectedLayer(in_channels, hidden_channels)
        self.fc2 = FullyConnectedLayer(hidden_channels, hidden_channels)
        self.linear = Linear(hidden_channels, 1)

    def forward(self, x):
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.linear(x)
        return x.squeeze(-1)

class Model(Module):
    def __init__(self):
        super().__init__()
        self.input = InputBlock()
        self.dense = DenseBlock(32)
        self.global_pool = GlobalPoolLayer(32 * K)
        self.output = OutputBlock(2 * 32 * K, 1024)

    def forward(self, input_batch):
        x = input_batch.x
        edge_index = input_batch.edge_index
        batch = input_batch.batch

        x, edge_index = self.input(x, edge_index)
        x = self.dense(x, edge_index)
        x = self.global_pool(x, batch)
        x = self.output(x)
        return x
