from .util import ENCODER_REGISTRY, DATASET_REGISTRY  # noqa

# Populates ENCODER_REGISTRY
from .encoders import DefaultEncoder  # noqa

# Populates DATASET_REGISTRY
from .dataset import (ARCStanceDataset,
                      RumourEvalTaskADataset,
                      DanishRumourDataset,
                      RussianStanceDataset)

from .datamodule import StanceDataModule  # noqa
