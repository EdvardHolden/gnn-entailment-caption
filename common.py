from torch.utils.data import DataLoader
from torch_geometric.data import Batch

BATCH_SIZE = 32

def mk_loader(data):
    return DataLoader(
        data,
        batch_size=BATCH_SIZE,
        collate_fn=Batch.from_data_list,
        shuffle=True,
        pin_memory=True,
        num_workers=8
    )
