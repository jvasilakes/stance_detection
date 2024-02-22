import random

import torch
import pytorch_lightning as pl
from torch.utils.data import DataLoader

from src.data import DATASET_REGISTRY


class StanceDataModule(pl.LightningDataModule):

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.dataset_cls = DATASET_REGISTRY[self.config.Data.dataset_name.value]
        self._ran_setup = False

    def setup(self, stage=None):
        self.dataset = self.dataset_cls.from_config(self.config)
        self.label_spec = self.dataset.label_spec
        self.batch_size = self.config.Training.batch_size.value
        try:
            self.tokenizer = self.dataset.encoder.tokenizer
        except AttributeError:
            self.tokenizer = None
        random.seed(self.config.Experiment.random_seed.value)
        torch.manual_seed(self.config.Experiment.random_seed.value)
        self._ran_setup = True

    def train_dataloader(self):
        if getattr(self.dataset.train, "__len__", None) is not None:
            random.shuffle(self.dataset.train)
        return DataLoader(self.dataset.train, batch_size=self.batch_size,
                          num_workers=4, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.dataset.val, batch_size=self.batch_size,
                          num_workers=4)

    def test_dataloader(self):
        return DataLoader(self.dataset.test, batch_size=self.batch_size,
                          num_workers=4)

    def __str__(self):
        return self.__class__.__name__

    def __repr__(self):
        return self.__class__.__name__
