from torch.utils.data import DataLoader
from torch_geometric.data import Batch

from dataset import DeepMathDataset, LTBDataset

# TODO this should be in a config file
BATCH_SIZE = 64


# TODO this should be removed and put in dataset.py
def mk_loader(root, name, batch_size=BATCH_SIZE, shuffle=True, **kwargs):
    dataset = DeepMathDataset(root, name=name)
    print(dataset)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        collate_fn=Batch.from_data_list,
        shuffle=shuffle,
        pin_memory=True,
        num_workers=8,
        **kwargs
    )


def mk_loader_ltb(root, name, caption=None, batch_size=BATCH_SIZE, shuffle=True, **kwargs):
    dataset = LTBDataset(root, name=name, caption=caption)
    print(dataset)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        collate_fn=Batch.from_data_list,
        shuffle=shuffle,
        pin_memory=True,
        num_workers=8,
        **kwargs
    )
