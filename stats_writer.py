from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
from collections import defaultdict
from typing import Dict, List
import pickle
import os

import config


class Writer:
    def __init__(self, model, file_prefix: str = ""):
        self.model = model
        self.step = 1
        self.file_prefix = file_prefix
        self.writer = self._get_summary_writer()
        self.scalar_dict = defaultdict(list)

    def _get_summary_writer(self):
        return SummaryWriter(
            os.path.join(config.dpath, ".log/" + self.file_prefix + datetime.now().strftime("%Y%m%d-%H%M%S"))
        )

    def save_scores(self, path):
        with open(os.path.join(path, "history.pkl"), "wb") as f:
            pickle.dump(self.get_scores(), f)

    def get_scores(self) -> Dict[str, List[float]]:
        return self.scalar_dict

    def on_step(self):
        self.writer.flush()
        self.step += 1

    def _writer_add_scalar(self, tag, scalar):
        self.writer.add_scalar(tag, scalar, global_step=self.step)
        self.scalar_dict[tag].append(scalar)

    def report_val_score(self, score):
        self._writer_add_scalar("score/val", score)

    def report_train_score(self, score):
        self._writer_add_scalar("score/train", score)

    def report_train_loss(self, loss):
        self._writer_add_scalar("loss/train", loss)

    def report_val_loss(self, loss):
        self._writer_add_scalar("loss/val", loss)

    def report_model_parameters(self):
        for name, parameter in self.model.named_parameters():
            key = "parameters/" + name.replace(".", "/")
            self.writer.add_histogram(key, parameter.data, global_step=self.step)

    def report_output(self, actual, predicted):
        self.writer.add_histogram("output/predicted", predicted, global_step=self.step)
        self.writer.add_histogram("output/error", predicted - actual, global_step=self.step)
