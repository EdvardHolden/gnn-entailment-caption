import torch
from torch.nn import BatchNorm1d, Embedding, Linear, Module, ModuleList
from torch.nn.functional import relu
from torch_geometric import nn as geom

import config

LAYERS = 24  # TODO what is this variable?
K = 8


class FullyConnectedLayer(Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.bn = BatchNorm1d(in_channels)
        self.linear = Linear(in_channels, out_channels)

    def forward(self, x):
        x = relu(x)
        x = self.bn(x)
        x = self.linear(x)
        return x


class ConvLayer(Module):
    def __init__(self):
        super().__init__()
        self.bn = BatchNorm1d(K)
        self.conv_in = geom.GCNConv(K, K)
        self.conv_out = geom.GCNConv(K, K, flow="target_to_source")

    def forward(self, x, edge_index):
        x = relu(x)
        x = self.bn(x)
        in_x = self.conv_in(x, edge_index)
        out_x = self.conv_out(x, edge_index)
        x = torch.cat((in_x, out_x), dim=1)
        return x


class DenseBlock(Module):
    def __init__(self, layers):
        super().__init__()
        self.fc = ModuleList([FullyConnectedLayer(2 * K * (layer + 1), K) for layer in range(layers)])
        self.conv = ModuleList([ConvLayer() for _ in range(layers)])

    def forward(self, x, edge_index):
        outputs = [x]
        for conv, fc in zip(self.conv, self.fc):
            combined = torch.cat(outputs, dim=1)
            x = fc(combined)
            x = conv(x, edge_index)
            outputs.append(x)

        return torch.cat(outputs, dim=1)


class GlobalPoolLayer(Module):
    def __init__(self, in_channels):
        super().__init__()
        self.bn = BatchNorm1d(in_channels)

    def forward(self, x, batch):
        x = self.bn(x)
        return geom.global_mean_pool(x, batch)


class Model(Module):
    def __init__(self):
        super().__init__()
        # TODO make separate value for embedding dimensions>
        self.input = Embedding(len(config.NODE_TYPE), 2 * K)
        self.dense = DenseBlock(LAYERS)
        self.output = FullyConnectedLayer(2 * (LAYERS + 1) * K, 1)

    def forward(self, input_batch):
        x = input_batch.x
        edge_index = input_batch.edge_index
        premise_index = input_batch.premise_index
        # batch = input_batch.batch

        # TODO why is global pool not used???

        x = self.input(x)
        emb = self.dense(x, edge_index)
        x_premise = emb[premise_index]
        x_premise = self.output(x_premise).squeeze(-1)

        return emb, x_premise
        # return x


def main():
    from torchinfo import summary

    model = Model()
    summary(model)
    print()
    print()
    # print(model)


if __name__ == "__main__":
    main()
