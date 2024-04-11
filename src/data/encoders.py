import os

import torch
from transformers import AutoTokenizer

from src.data.util import register_encoder


os.environ["TOKENIZERS_PARALLELISM"] = "false"


class Encoder(object):

    @classmethod
    def from_config(cls, config):
        return cls(config.Data.Encoder.pretrained_model_name_or_path.value,
                   config.Data.Encoder.max_seq_length.value)

    def __init__(self, pretrained_model_name_or_path="bert-base-uncased",
                 max_seq_length=256, cache=True):
        self.pretrained_model_name_or_path = pretrained_model_name_or_path
        self.max_seq_length = max_seq_length
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.pretrained_model_name_or_path)
        # Whether to cache encoded examples
        self.cache = cache
        self._cache = {}

    def __call__(self, examples):
        if isinstance(examples, list):
            return [self._encode_and_cache(example)
                    for example in examples if example is not None]
        else:
            # There is just a single example
            return self._encode_and_cache(examples)

    def encode_single_example(self, example):
        raise NotImplementedError

    def _encode_and_cache(self, example):
        key = example["__key__"]
        try:
            return self._cache[key]
        except KeyError:
            encoded = self.encode_single_example(example)
            if self.cache is True:
                self._cache[key] = encoded
            return encoded


@register_encoder("default")
class DefaultEncoder(Encoder):

    def run_tokenize(self, example):
        data = example["json"]
        text_pair = [data["target"], data["body"]]
        encoded = self.tokenizer([text_pair], max_length=self.max_seq_length,
                                 padding="max_length", truncation=True,
                                 return_tensors="pt")
        return encoded

    def encode_single_example(self, example):
        encoded = self.run_tokenize(example)
        # squeeze because the tokenizer always includes a batch dimension,
        # which we don't want quite yet.
        example["json"]["encoded"] = {k: v.squeeze() for (k, v) in encoded.items()}  # noqa
        return example


@register_encoder("default-t5")
class DefaultEncoderT5(DefaultEncoder):

    def run_tokenize(self, example):
        data = example["json"]
        inputs = f"claim: {data['target']} stance: {data['body']}"
        encoded = self.tokenizer(
            inputs, max_length=self.max_seq_length,
            padding="max_length", truncation=True, return_tensors="pt")
        return encoded


@register_encoder("default-t5-gen")
class DefaultEncoderT5Generative(DefaultEncoder):

    def run_tokenize(self, example):
        data = example["json"]
        inputs = f"claim: {data['target']} stance: {data['body']}"
        inputs_encoded = self.tokenizer(
                inputs, max_length=self.max_seq_length,
                padding="max_length", truncation=True, return_tensors="pt")
        label_text = data["labels"]["Stance"]  # TODO: don't hard code
        labels_encoded = self.tokenizer(
                label_text, max_length=self.max_seq_length,
                padding="max_length", truncation=True, return_tensors="pt")
        encoded = {"input_ids": inputs_encoded["input_ids"],
                   "decoder_input_ids": labels_encoded["input_ids"]}
        return encoded


@register_encoder("directional-attention")
class DirectionalAttentionEncoder(Encoder):

    def run_tokenize(self, example):
        data = example["json"]
        text_pair = [data["target"], data["body"]]
        encoded = self.tokenizer([text_pair], max_length=self.max_seq_length,
                                 padding="max_length", truncation=True,
                                 return_tensors="pt")
        return encoded

    def encode_single_example(self, example):
        encoded = self.run_tokenize(example)
        if self.tokenizer.sep_token_id is None:
            raise AttributeError("DirectionalAttentionEncoder assumes target and body are concatenated, which does not seem to be the case.")  # noqa
        # BERT
        # Get index of first occurrence of [SEP] token
        # Add 1 because we want to include the [SEP] token
        target_seq_length = 1 + (encoded["input_ids"][0] == self.tokenizer.sep_token_id).nonzero()[0].item()  # noqa
        unpadded_seq_length = encoded["attention_mask"].sum()

        attn_matrix = torch.zeros((self.max_seq_length, self.max_seq_length),
                                  dtype=torch.long)

        # self attention on target
        # Directional Attention Matrix
        # Rows are the "start" tokens, columns are the "end" tokens.
        # So, all "start" tokens can see the target tokens.
        attn_matrix[:unpadded_seq_length, :target_seq_length] = 1
        # and the body tokens can see each other and the target tokens
        attn_matrix[target_seq_length:unpadded_seq_length, :unpadded_seq_length] = 1  # noqa
        encoded["attention_mask"] = attn_matrix

        # squeeze because the tokenizer always includes a batch dimension,
        # which we don't want quite yet.
        example["json"]["encoded"] = {k: v.squeeze() for (k, v) in encoded.items()}  # noqa
        return example
