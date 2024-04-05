import warnings
from copy import deepcopy
from collections import defaultdict

import torch
import torch.nn as nn
import numpy as np
import pytorch_lightning as pl
from sklearn.metrics import precision_recall_fscore_support
from transformers import AutoConfig, AutoModel
from transformers import logging as transformers_logging
from transformers.modeling_outputs import SequenceClassifierOutput

from src.modeling.util import register_model
from src.modeling import TOKEN_POOLER_REGISTRY


# Ignore warning that AutoModel is not using some parameters.
transformers_logging.set_verbosity_error()


@register_model("stance-pooling")
class StancePoolingModel(pl.LightningModule):

    def __init__(self, config, label_spec):
        super().__init__()
        self.config = config
        self.model = LLMForMultiTaskSequenceClassificationWithPooling.from_config(  # noqa
            config, label_spec)
        self.validation_step_outputs = []

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
        output = {"__key__": batch["__key__"],
                  "loss": total_loss.detach().cpu().item(),
                  "task_losses": task_losses,
                  "task_logits": task_logits,
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
        self.log("avg_val_f1_total", np.mean(all_f1s))
        self.validation_step_outputs.clear()

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


class LLMForMultiTaskSequenceClassificationWithPooling(nn.Module):

    @classmethod
    def from_config(cls, config, label_spec):
        return cls(config.Model.pretrained_model_name_or_path.value,
                   config.Model.body_pool_fn.value,
                   label_spec,
                   dropout_prob=config.Model.dropout_prob.value,
                   freeze_pretrained=config.Model.freeze_pretrained.value)

    def __init__(self, pretrained_model_name_or_path, body_pool_fn,
                 label_spec, dropout_prob=0.0, freeze_pretrained=False):
        super().__init__()
        self.pretrained_model_name_or_path = pretrained_model_name_or_path
        self.body_pool_fn = body_pool_fn
        self.label_spec = label_spec
        self.dropout_prob = dropout_prob
        self.freeze_pretrained = freeze_pretrained

        self.llm_config = AutoConfig.from_pretrained(
            self.pretrained_model_name_or_path)
        self.llm = AutoModel.from_pretrained(
            self.pretrained_model_name_or_path, config=self.llm_config)

        if self.freeze_pretrained is True:
            for param in self.llm.parameters():
                param.requires_grad = False

        self.classifier_heads = nn.ModuleDict()
        self.body_pool_fns = nn.ModuleDict()
        classifier_insize = self.llm_config.hidden_size
        body_pool_cls = TOKEN_POOLER_REGISTRY[self.body_pool_fn]
        for (task, labeldim) in self.label_spec.items():
            self.body_pool_fns[task] = body_pool_cls(
                classifier_insize, classifier_insize)
            self.classifier_heads[task] = nn.Sequential(
                nn.Dropout(self.dropout_prob),
                nn.Linear(classifier_insize, labeldim)
            )
        self.loss_fn = nn.CrossEntropyLoss()

    def forward(self, llm_inputs, labels=None):
        llm_outputs = self.llm(**llm_inputs,
                               return_dict=True)

        # Mask out everything but the body text
        token_mask = torch.zeros_like(llm_outputs.last_hidden_state)
        if "token_type_ids" in llm_inputs.keys():  # BERT
            token_mask[llm_inputs["token_type_ids"] == 1] = 1
        elif "decoder_input_ids" in llm_inputs.keys():  # T5
            token_mask[llm_inputs["decoder_input_ids"] != 0] = 1
        else:
            raise AssertionError("Need token_type_ids or decoder_input_ids to create a token mask!")  # noqa

        clf_outputs = {}
        for (task, clf_head) in self.classifier_heads.items():
            pooled_output = self.body_pool_fns[task](
                llm_outputs.last_hidden_state, token_mask)
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
