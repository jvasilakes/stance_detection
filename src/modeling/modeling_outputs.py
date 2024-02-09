from typing import Optional, Tuple

import torch
from dataclasses import dataclass
from transformers.file_utils import ModelOutput


@dataclass
class SequenceClassifierOutputWithTokenMask(ModelOutput):

    logits: torch.FloatTensor = None
    loss: Optional[torch.FloatTensor] = None
    mask_loss: Optional[torch.FloatTensor] = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None
    mask: Optional[torch.FloatTensor] = None
