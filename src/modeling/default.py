import warnings
from copy import deepcopy
from collections import defaultdict

import torch
import torch.nn as nn
import numpy as np
import pytorch_lightning as pl
from sklearn.metrics import precision_recall_fscore_support
from transformers import BertConfig, BertModel
from transformers import logging as transformers_logging
from transformers.modeling_outputs import SequenceClassifierOutput

from src.modeling.util import register_model


# Ignore warning that BertModel is not using some parameters.
transformers_logging.set_verbosity_error()


@register_model("default")
class StanceModelModule(pl.LightningModule):

    def __init__(self, config, label_spec):
        super().__init__()
        self.config = config
        self.model = BertForMultiTaskSequenceClassification.from_config(
            config, label_spec)

    def get_model_outputs(self, batch):
        try:
            labels = batch["json"]["labels"]
        except KeyError:
            labels = None
        return self(batch["json"]["encoded"], labels=labels)

    def forward(self, inputs, labels=None):
        return self.model(inputs, labels=labels)

    def training_step(self, batch, batch_idx):
        outputs = self.get_model_outputs(batch)
        total_loss = torch.tensor(0.0).to(self.device)
        for (task, task_out) in outputs.items():
            loss = task_out.loss
            total_loss += task_out.loss
            self.log(f"train_loss_{task}", loss.detach().cpu().item())
        self.log("train_loss_total", total_loss.detach().cpu().item())
        return {"__key__": batch["__key__"],
                "loss": total_loss}

    def validation_step(self, batch, batch_idx):
        outputs = self.get_model_outputs(batch)
        total_loss = torch.tensor(0.0).to(self.device)
        task_losses = {}
        task_logits = {}
        for (task, task_out) in outputs.items():
            loss = task_out.loss
            total_loss += task_out.loss
            task_losses[task] = loss.detach().cpu().item()
            task_logits[task] = task_out.logits.detach().cpu()
        return {"__key__": batch["__key__"],
                "loss": total_loss.detach().cpu().item(),
                "task_losses": task_losses,
                "task_logits": task_logits,
                "task_labels": batch["json"]["labels"]}

    def validation_epoch_end(self, batch_outputs):
        total_losses = []
        all_task_losses = defaultdict(list)
        all_preds = defaultdict(list)
        all_labels = defaultdict(list)
        for batch in batch_outputs:
            total_losses.append(batch["loss"])
            for (task, task_loss) in batch["task_losses"].items():
                all_task_losses[task].append(task_loss)
                logits = batch["task_logits"][task]
                preds = self.model.predict_from_logits(task, logits)
                all_preds[task].extend(preds.detach().cpu().numpy())
                all_labels[task].extend(batch["task_labels"][task].detach().cpu().numpy())  # noqa
        self.log("avg_val_loss_total", np.mean(total_losses))

        all_f1s = []
        for (task, losses) in all_task_losses.items():
            self.log(f"avg_val_loss_{task}", np.mean(losses))

            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                _, _, task_f1, _ = precision_recall_fscore_support(
                    all_labels[task], all_preds[task], average="macro")
            all_f1s.append(task_f1)
        self.log("avg_val_f1", np.mean(all_f1s))

    def predict_step(self, batch, batch_idx):
        outputs = self.get_model_outputs(batch)
        batch_cp = deepcopy(batch)
        batch_cp["json"]["predictions"] = {}
        for (task, task_out) in outputs.items():
            logits = task_out.logits
            preds = self.model.predict_from_logits(task, logits)
            batch_cp["json"]["predictions"][task] = preds
        return batch_cp

    def configure_optimizers(self):
        lr = self.config.Training.learn_rate.value
        weight_decay = self.config.Training.weight_decay.value
        opt = torch.optim.AdamW(self.parameters(), lr=lr,
                                weight_decay=weight_decay)
        return [opt]


class BertForMultiTaskSequenceClassification(nn.Module):

    @classmethod
    def from_config(cls, config, label_spec):
        return cls(config.Model.pretrained_model_name_or_path.value,
                   label_spec,
                   dropout_prob=config.Model.dropout_prob.value)

    def __init__(self, pretrained_model_name_or_path, label_spec,
                 dropout_prob=0.0):
        super().__init__()
        self.pretrained_model_name_or_path = pretrained_model_name_or_path
        self.label_spec = label_spec
        self.dropout_prob = dropout_prob

        self.bert_config = BertConfig.from_pretrained(
            self.pretrained_model_name_or_path)
        # Can override BERT config values here, e.g., dropout.
        self.bert = BertModel.from_pretrained(
            self.pretrained_model_name_or_path, config=self.bert_config)

        self.classifier_heads = nn.ModuleDict()
        classifier_insize = self.bert_config.hidden_size
        for (task, labeldim) in self.label_spec.items():
            self.classifier_heads[task] = nn.Sequential(
                nn.Dropout(self.dropout_prob),
                nn.Linear(classifier_insize, labeldim)
            )
        self.loss_fn = nn.CrossEntropyLoss()

    def forward(self, bert_inputs, labels=None):
        bert_outputs = self.bert(**bert_inputs,
                                 return_dict=True)
        pooled_output = bert_outputs.pooler_output
        clf_outputs = {}
        for (task, clf_head) in self.classifier_heads.items():
            logits = clf_head(pooled_output)
            if labels is not None:
                clf_loss = self.loss_fn(logits.view(-1, self.label_spec[task]),
                                        labels[task].view(-1))
            else:
                clf_loss = None
            clf_outputs[task] = SequenceClassifierOutput(
                loss=clf_loss, logits=logits)
        return clf_outputs

    def predict_from_logits(self, task, logits):
        labeldim = self.label_spec[task]
        if labeldim == 1:
            preds = (logits.sigmoid() >= 0.5).long()
        else:
            preds = logits.argmax(1)
        return preds
