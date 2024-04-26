import warnings
from copy import deepcopy
from collections import defaultdict

import torch
import torch.nn as nn
import numpy as np
import pytorch_lightning as pl
from sklearn.metrics import precision_recall_fscore_support
from transformers import AutoConfig, T5ForConditionalGeneration, AutoTokenizer
from transformers import logging as transformers_logging

from src.modeling.util import register_model
from src.modeling.modeling_mt5 import MT5ForExplainableConditionalGeneration


# Ignore warning that AutoModel is not using some parameters.
transformers_logging.set_verbosity_error()


@register_model("default-t5-gen")
class StanceModelT5Gen(pl.LightningModule):

    def __init__(self, config, label_spec):
        super().__init__()
        self.config = config
        self.tokenizer = AutoTokenizer.from_pretrained(
                config.Data.Encoder.pretrained_model_name_or_path.value)
        self.model = T5ForSequenceClassification.from_config(
            config, label_spec, self.tokenizer)
        self.validation_step_outputs = []

    def get_model_outputs(self, batch):
        return self(batch["json"]["encoded"])

    def forward(self, inputs):
        return self.model(inputs)

    def training_step(self, batch, batch_idx):
        outputs = self.get_model_outputs(batch)
        total_loss = outputs.loss
        self.log("train_loss_total", total_loss.detach().cpu().item())
        return {"__key__": batch["__key__"],
                "loss": total_loss}

    def validation_step(self, batch, batch_idx):
        outputs = self.get_model_outputs(batch)
        output = {"__key__": batch["__key__"],
                  "input_ids": batch["json"]["encoded"]["input_ids"],
                  "loss": outputs.loss.detach().cpu().item(),
                  "task_losses": {"Stance": outputs.loss.detach().cpu().item()},  # noqa
                  "task_logits": {"Stance": outputs.logits.detach()},
                  "task_labels": batch["json"]["labels"]}
        self.validation_step_outputs.append(output)

    def on_validation_epoch_end(self):
        total_losses = []
        all_task_losses = defaultdict(list)
        all_preds = defaultdict(list)
        all_labels = defaultdict(list)
        for batch in self.validation_step_outputs:
            total_losses.append(batch["loss"])
            for (task, task_loss) in batch["task_losses"].items():
                all_task_losses[task].append(task_loss)
                logits = batch["task_logits"][task]
                preds = self.model.predict_from_logits(logits)
                preds_text = self.tokenizer.batch_decode(
                        preds, skip_special_tokens=True)
                all_preds[task].extend(preds_text)
                labs = batch["task_labels"][task]
                if torch.is_tensor(labs):
                    labs = batch["task_labels"][task].detach().cpu().numpy()
                all_labels[task].extend(labs)
        self.log("avg_val_loss_total", np.mean(total_losses))

        all_f1s = []
        for (task, losses) in all_task_losses.items():
            self.log(f"avg_val_loss_{task}", np.mean(losses))

            label_set = list(set(all_labels[task]))
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                _, _, task_f1, _ = precision_recall_fscore_support(
                    all_labels[task], all_preds[task], average="macro",
                    labels=label_set)
            all_f1s.append(task_f1)
        self.log("avg_val_f1_total", np.mean(all_f1s))
        self.validation_step_outputs.clear()

    def predict_step(self, batch, batch_idx):
        batch_cp = deepcopy(batch)
        batch_cp["json"]["predictions"] = {}
        outputs = self.get_model_outputs(batch)
        preds = self.model.predict_from_logits(outputs.logits)
        preds = self.tokenizer.batch_decode(preds, skip_special_tokens=True)
        batch_cp["json"]["predictions"]["Stance"] = preds
        cross_attns = outputs.cross_attentions[-1][:, -1, :, :].detach().cpu().tolist()
        batch_cp["json"]["cross_attns"] = cross_attns
        return batch_cp

    def configure_optimizers(self):
        lr = self.config.Training.learn_rate.value
        weight_decay = self.config.Training.weight_decay.value
        opt = torch.optim.AdamW(self.parameters(), lr=lr,
                                weight_decay=weight_decay)
        return [opt]


class T5ForSequenceClassification(nn.Module):

    @classmethod
    def from_config(cls, config, label_spec, tokenizer):
        return cls(config.Model.pretrained_model_name_or_path.value,
                   label_spec, tokenizer,
                   label_weights=config.Training.label_weights.value,
                   dropout_prob=config.Model.dropout_prob.value,
                   freeze_pretrained=config.Model.freeze_pretrained.value)

    def __init__(self, pretrained_model_name_or_path, label_spec, tokenizer,
                 label_weights=None, dropout_prob=0.0, freeze_pretrained=False):  # noqa
        super().__init__()
        self.pretrained_model_name_or_path = pretrained_model_name_or_path
        if len(label_spec.keys()) > 1:
            raise ValueError("T5 model can only handle single-task classification.")  # noqa
        self.label_spec = label_spec
        self.tokenizer = tokenizer
        self.label_weights_dict = label_weights or {}
        self.dropout_prob = dropout_prob
        self.freeze_pretrained = freeze_pretrained

        self.llm_config = AutoConfig.from_pretrained(
            self.pretrained_model_name_or_path)
        # Can override LLM config values here, e.g., dropout.
        #self.llm = T5ForConditionalGeneration.from_pretrained(
        self.llm = MT5ForExplainableConditionalGeneration.from_pretrained(
            self.pretrained_model_name_or_path, config=self.llm_config,
            torch_dtype=torch.bfloat16)

        # Align label weights to tokenized labels
        # Because this is T5, encode_labels=False and so label_spec is
        # a dictionary of {task: {label: int}}
        label_texts = [label for (task, labels) in self.label_spec.items()
                       for label in labels.keys()]
        label_texts.append("</s>")
        label_tok_ids = tokenizer(
            label_texts, add_special_tokens=False)["input_ids"]
        # {label: [token_ids]}
        labels2tokids = dict(zip(label_texts, label_tok_ids))
        task = list(self.label_spec.keys())[0]
        self.label_weights_dict[task]["</s>"] = 1.
        label_weights = self._get_label_weights(
                task, self.label_weights_dict, labels2tokids)
        self.loss_fn = nn.CrossEntropyLoss(
            weight=label_weights, ignore_index=-100)

        # Flatten and convert to tensor so it can be used to index logits.
        # Used by generate() in predict_from_input_ids()
        self.label_tok_ids = torch.as_tensor(
            [tok_id for lab_ids in label_tok_ids for tok_id in lab_ids])
        self.label_fn = lambda *args: self.label_tok_ids

        if self.freeze_pretrained is True:
            for param in self.llm.parameters():
                param.requires_grad = False

    def _get_label_weights(self, task, label_weights_dict, labels2tokids):
        # We need to create a list of label weights that is
        # aligned with the label tokens, since tokenization can split
        # labels into multiple tokens.
        label_weights = torch.zeros(self.llm.config.vocab_size,
                                    dtype=self.llm.dtype)
        try:
            task_weights = label_weights_dict[task]
        except KeyError:
            task_weights = {lab: 1. for lab in labels2tokids.keys()}
        for (lab, tok_ids) in labels2tokids.items():
            weight = task_weights[lab]
            for tok_id in tok_ids:
                label_weights[tok_id] = weight
        return label_weights

    def forward(self, llm_inputs):
        llm_outputs = self.llm(**llm_inputs, return_dict=True, output_attentions=True)
        if "labels" in llm_inputs.keys():
            weighted_loss = self.compute_loss_from_logits(
                    llm_outputs.logits, llm_inputs["labels"])
            llm_outputs.loss = weighted_loss
        return llm_outputs

    def compute_loss_from_logits(self, logits, labels):
        return self.loss_fn(logits.view(-1, logits.size(-1)), labels.view(-1))

    def predict_from_input_ids(self, input_ids):
        # TODO: For some reason using generate to predict performs much
        # worse than predicting directly from logits as below.
        return self.llm.generate(input_ids, max_new_tokens=3,
                                 prefix_allowed_tokens_fn=self.label_fn)

    def predict_from_logits(self, logits):
        lab_ids = self.label_tok_ids.to(logits.device)
        idxs = logits[:, :, lab_ids].argmax(-1)
        return lab_ids[idxs]
