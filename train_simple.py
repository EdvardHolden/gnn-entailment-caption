from pathlib import Path
import torch
from torch.nn.functional import binary_cross_entropy_with_logits
from torch.optim import SGD
from torch.optim.lr_scheduler import CyclicLR
from tqdm import tqdm

import config
from dataset import get_data_loader
from model_old import Model
from stats_writer import Writer


def batch_loss(model, batch):
    batch = batch.to(config.device)
    _, y = model(batch)
    loss = binary_cross_entropy_with_logits(y, batch.y)
    del batch
    return y, loss


def validation_loss(model, validation):
    losses = []
    with torch.no_grad():
        for batch in tqdm(validation):
            _, loss = batch_loss(model, batch)
            losses.append(loss.mean())

    return torch.tensor(losses).mean()


def train():

    ids = "id_files/train_256_rnd.txt"
    train_data = get_data_loader(ids)
    val_data = get_data_loader(ids)

    model = Model().to(config.device)
    optimizer = SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=1e-4, nesterov=True)
    scheduler = CyclicLR(optimizer, 0.01, 0.1, mode="exp_range", gamma=0.99995, step_size_up=4000)

    stats = Writer(model)
    best_loss = torch.tensor(float("inf"))

    while True:
        #stats.report_model_parameters()

        print("validating...")
        model.eval()
        val_loss = validation_loss(model, val_data)
        stats.report_val_loss(val_loss)
        if val_loss < best_loss:
            torch.save(model.state_dict(), "model.pt")
            best_loss = val_loss
        print(f"...done, loss {val_loss:.3E}")

        model.train()
        for batch in tqdm(train_data):
            optimizer.zero_grad()
            y, loss = batch_loss(model, batch)
            loss.backward()
            optimizer.step()
            scheduler.step()

            stats.report_train_loss(loss.mean())
            #if stats.step % 32 == 0:
            #    stats.report_output(batch.y, torch.sigmoid(y))
            stats.on_step()


if __name__ == "__main__":
    torch.manual_seed(0)
    try:
        train()
    except KeyboardInterrupt:
        pass
