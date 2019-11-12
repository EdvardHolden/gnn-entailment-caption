from torch.nn import BatchNorm1d, Linear, Module, ModuleList
from torch.nn.functional import dropout, relu
from torch_geometric.nn import GCNConv, EdgePooling, global_max_pool
from torch_geometric.utils import to_undirected

DROPOUT = 0.1

class EmbedBlock(Module):
    def __init__(self, channels):
        super().__init__()
        self.conv = GCNConv(4, channels)

    def forward(self, x, edge_index):
        x = self.conv(x, edge_index)
        return x

class ConvBlock(Module):
    def __init__(self, channels_in, channels_out):
        super().__init__()
        self.bn = BatchNorm1d(channels_in)
        self.conv = GCNConv(channels_in, channels_out)

    def forward(self, x, edge_index):
        x = self.bn(x)
        x = self.conv(x, edge_index)
        x = relu(x)
        return x

class PoolBlock(Module):
    def __init__(self, channels):
        super().__init__()
        self.bn = BatchNorm1d(channels)
        self.pool = EdgePooling(channels, dropout=DROPOUT)

    def forward(self, x, edge_index, batch):
        x = self.bn(x)
        x, edge_index, batch, unpool = self.pool(x, edge_index, batch)
        return (x, edge_index, batch)

class ConvPoolBlock(Module):
    def __init__(self, channels):
        super().__init__()
        self.conv = ModuleList([
            ConvBlock(channels, channels)
            for _ in range(CONV_PER_POOL)
        ])
        self.pool = PoolBlock(channels)

    def forward(self, x, edge_index, batch):
        for conv in self.conv:
            x = conv(x, edge_index)

        x, edge_index, batch = self.pool(x, edge_index, batch)
        return (x, edge_index, batch)

class FullyConnectedBlock(Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.bn = BatchNorm1d(in_channels)
        self.linear = Linear(in_channels, out_channels)

    def forward(self, x):
        x = self.bn(x)
        x = dropout(x, p=DROPOUT)
        x = self.linear(x)
        x = relu(x)
        return x

class GlobalPoolBlock(Module):
    def __init__(self, channels, reduction='max'):
        super().__init__()
        self.bn = BatchNorm1d(channels)
        self.op = {'max': global_max_pool}[reduction]

    def forward(self, x, batch):
        x = self.bn(x)
        x = self.op(x, batch)
        x = relu(x)
        return x

class OutputBlock(Module):
    def __init__(self, channels):
        super().__init__()
        self.bn = BatchNorm1d(channels)
        self.linear = Linear(channels, 1)

    def forward(self, x):
        x = self.bn(x)
        x = self.linear(x)
        return x.squeeze(-1)

class SimpleModel(Module):
    def __init__(self):
        super().__init__()
        self.embed = EmbedBlock(128)
        self.global_pool = GlobalPoolBlock(128)
        self.fc = FullyConnectedBlock(128, 128)
        self.output = OutputBlock(128)

    def forward(self, input_batch):
        x = input_batch.x
        edge_index = to_undirected(input_batch.edge_index)
        batch = input_batch.batch

        x = self.embed(x, edge_index)
        x = self.global_pool(x, batch)
        x = self.fc(x)
        x = self.output(x)
        return x
