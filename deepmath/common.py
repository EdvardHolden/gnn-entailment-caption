from torch.utils.data import DataLoader
from torch_geometric.data import Batch

from dataset import DeepMathDataset

BATCH_SIZE = 64

def mk_loader(root, name, **kwargs):
    dataset = DeepMathDataset(root, name=name)
    return DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        collate_fn=Batch.from_data_list,
        shuffle=True,
        pin_memory=True,
        num_workers=8,
        **kwargs
    )
