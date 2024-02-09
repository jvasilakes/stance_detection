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

from src.modeling import TOKEN_POOLER_REGISTRY
from src.modeling.util import register_model
from src.modeling.modeling_outputs import SequenceClassifierOutputWithTokenMask


# Ignore warning that BertModel is not using some parameters.
transformers_logging.set_verbosity_error()


@register_model("stance-pooling-attention")
class StancePoolingModelWithAttention(pl.LightningModule):

    def __init__(self, config, label_spec):
        super().__init__()
        self.config = config
        self.model = BertForMultiTaskSequenceClassificationWithAttentionPooling.from_config(  # noqa
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
            self.log(f"step_train_loss_{task}", loss.detach().cpu().item())
        self.log("step_train_loss_total", total_loss.detach().cpu().item())
        return {"__key__": batch["__key__"],
                "loss": total_loss}

    def validation_step(self, batch, batch_idx):
        outputs = self.get_model_outputs(batch)
        total_loss = torch.tensor(0.0).to(self.device)
        task_losses = {}
        task_logits = {}

        mask_coverages = {}
        mask_vals = {}
        # Used for computing mask coverage
        input_ids = batch["json"]["encoded"]["input_ids"]
        token_type_ids = batch["json"]["encoded"]["token_type_ids"]
        body_tokens = input_ids * token_type_ids
        seq_lengths = torch.logical_not(body_tokens == 0).sum(1)

        for (task, task_out) in outputs.items():
            loss = task_out.loss
            total_loss += task_out.loss
            task_losses[task] = loss.detach().cpu().item()
            task_logits[task] = task_out.logits.detach().cpu()
            mask_coverage = (task_out.mask > 0).sum(1) / seq_lengths
            mask_coverages[task] = mask_coverage.detach().cpu().numpy()
            mask_vals[task] = task_out.mask[task_out.mask > 0].detach().cpu().numpy()  # noqa
        return {"__key__": batch["__key__"],
                "loss": total_loss.detach().cpu().item(),
                "task_losses": task_losses,
                "task_logits": task_logits,
                "task_labels": batch["json"]["labels"],
                "mask_coverage": mask_coverages,
                "mask_vals": mask_vals}

    def validation_epoch_end(self, batch_outputs):
        total_losses = []
        all_task_losses = defaultdict(list)
        all_preds = defaultdict(list)
        all_labels = defaultdict(list)
        all_mask_coverages = defaultdict(list)
        all_mask_vals = defaultdict(list)
        for batch in batch_outputs:
            total_losses.append(batch["loss"])
            for (task, task_loss) in batch["task_losses"].items():
                all_task_losses[task].append(task_loss)
                logits = batch["task_logits"][task]
                preds = self.model.predict_from_logits(task, logits)
                all_preds[task].extend(preds.detach().cpu().numpy())
                all_labels[task].extend(batch["task_labels"][task].detach().cpu().numpy())  # noqa
                all_mask_coverages[task].extend(batch["mask_coverage"][task])
                all_mask_vals[task].extend(batch["mask_vals"][task])
        self.log("avg_val_loss_total", np.mean(total_losses))

        all_f1s = []
        for (task, losses) in all_task_losses.items():
            self.log(f"avg_val_loss_{task}", np.mean(losses))
            self.log(f"avg_mask_coverage_{task}",
                     np.mean(all_mask_coverages[task]))
            self.log(f"avg_mask_mean_{task}", np.mean(all_mask_vals[task]))
            self.log(f"avg_mask_sd_{task}", np.std(all_mask_vals[task]))

            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                _, _, task_f1, _ = precision_recall_fscore_support(
                    all_labels[task], all_preds[task], average="macro")
            all_f1s.append(task_f1)
        self.log("avg_val_f1_total", np.mean(all_f1s))

    def predict_step(self, batch, batch_idx):
        outputs = self.get_model_outputs(batch)
        batch_cp = deepcopy(batch)
        batch_cp["json"]["predictions"] = {}
        batch_cp["json"]["token_masks"] = {}
        for (task, task_out) in outputs.items():
            logits = task_out.logits
            preds = self.model.predict_from_logits(task, logits)
            batch_cp["json"]["predictions"][task] = preds
            batch_cp["json"]["token_masks"][task] = task_out.mask
        return batch_cp

    def configure_optimizers(self):
        lr = self.config.Training.learn_rate.value
        weight_decay = self.config.Training.weight_decay.value
        opt = torch.optim.AdamW(self.parameters(), lr=lr,
                                weight_decay=weight_decay)
        return [opt]


class BertForMultiTaskSequenceClassificationWithAttentionPooling(nn.Module):

    @classmethod
    def from_config(cls, config, label_spec):
        return cls(config.Model.pretrained_model_name_or_path.value,
                   config.Model.target_pool_fn.value,
                   config.Model.body_projection_fn_kwargs.value,
                   config.Model.body_pool_fn.value,
                   label_spec,
                   dropout_prob=config.Model.dropout_prob.value,
                   freeze_pretrained=config.Model.freeze_pretrained.value)

    def __init__(self, pretrained_model_name_or_path,
                 target_pool_fn, body_projection_fn_kwargs,
                 body_pool_fn, label_spec, dropout_prob=0.0,
                 freeze_pretrained=False):
        super().__init__()
        self.pretrained_model_name_or_path = pretrained_model_name_or_path
        self.target_pool_fn = target_pool_fn
        self.body_projection_fn_kwargs = body_projection_fn_kwargs
        self.body_pool_fn = body_pool_fn
        self.label_spec = label_spec
        self.dropout_prob = dropout_prob
        self.freeze_pretrained = freeze_pretrained

        self.bert_config = BertConfig.from_pretrained(
            self.pretrained_model_name_or_path)
        self.bert = BertModel.from_pretrained(
            self.pretrained_model_name_or_path, config=self.bert_config)

        if self.freeze_pretrained is True:
            for param in self.bert.parameters():
                param.requires_grad = False

        target_pool_cls = TOKEN_POOLER_REGISTRY[self.target_pool_fn]
        self.target_pooler = target_pool_cls(
            self.bert_config.hidden_size, self.bert_config.hidden_size)

        self.classifier_heads = nn.ModuleDict()
        self.body_poolers = nn.ModuleDict()
        classifier_insize = self.bert_config.hidden_size
        body_pool_cls = TOKEN_POOLER_REGISTRY[self.body_pool_fn]
        for (task, labeldim) in self.label_spec.items():
            self.body_poolers[task] = body_pool_cls(
                classifier_insize, classifier_insize,
                **self.body_projection_fn_kwargs)
            self.classifier_heads[task] = nn.Sequential(
                nn.Dropout(self.dropout_prob),
                nn.Linear(classifier_insize, labeldim)
            )
        self.loss_fn = nn.CrossEntropyLoss()

    def forward(self, bert_inputs, labels=None):
        bert_outputs = self.bert(**bert_inputs,
                                 return_dict=True)

        if bert_inputs["attention_mask"].dim() == 2:
            # Attention vector. The target mask is thus where the
            # token_type_ids are 0 and the attention_mask is 1.
            target_mask = torch.logical_xor(bert_inputs["token_type_ids"],
                                            bert_inputs["attention_mask"]).int()  # noqa
        else:
            # Attention matrix. Because the target only attends to itself
            # the target mask is simply the first row of the attention_mask.
            target_mask = bert_inputs["attention_mask"][:, 0]

        # TODO: XGLM has no pooler_output
        bert_hidden_dim = bert_outputs.pooler_output.size(-1)
        target_mask = target_mask.unsqueeze(-1).repeat(1, 1, bert_hidden_dim)

        body_mask = torch.zeros_like(bert_outputs.last_hidden_state)
        # TODO: XGLM has no token_type_ids
        body_mask[bert_inputs["token_type_ids"] == 1] = 1

        target_pooled = self.target_pooler(bert_outputs.last_hidden_state,
                                           target_mask)

        clf_outputs = {}
        for (task, clf_head) in self.classifier_heads.items():
            pooled_output, attentions = self.body_poolers[task](
                bert_outputs.last_hidden_state, body_mask, target_pooled)
            logits = clf_head(pooled_output)
            if labels is not None:
                clf_loss = self.loss_fn(logits.view(-1, self.label_spec[task]),
                                        labels[task].view(-1))
            else:
                clf_loss = None
            clf_outputs[task] = SequenceClassifierOutputWithTokenMask(
                loss=clf_loss, logits=logits, mask=attentions.squeeze(2))
        return clf_outputs

    def predict_from_logits(self, task, logits):
        labeldim = self.label_spec[task]
        if labeldim == 1:
            preds = (logits.sigmoid() >= 0.5).long()
        else:
            preds = logits.argmax(1)
        return preds
