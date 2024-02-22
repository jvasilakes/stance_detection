import os
import sys
import torch
from transformers import BertConfig, BertTokenizer, BertModel
from transformers.models.bert.modeling_bert import BertEmbeddings, BertSelfAttention  # noqa

from pytorch_lightning import seed_everything

sys.path.insert(0, os.path.abspath(os.getcwd()))
from src.data.encoders import DirectionalAttentionEncoder
from config import config


MAX_LENGTH = 15


"""
This script tests that a Bert Model with directional attention mask
encodes the target sequence identically as 1) a model with no body text
and 2) input with a different body.
"""


class TestModel(torch.nn.Module):

    def __init__(self, bert_config):
        super().__init__()
        self.bert = BertModel(bert_config)
        self.emb_layer = BertEmbeddings(bert_config)
        self.attn_layer = BertSelfAttention(bert_config)

    def forward(self,
                input_ids=None,
                attention_mask=None,
                token_type_ids=None,
                position_ids=None,
                head_mask=None,
                inputs_embeds=None,
                encoder_hidden_states=None,
                encoder_attention_mask=None,
                past_key_values=None,
                use_cache=None,
                output_attentions=None,
                output_hidden_states=None,
                return_dict=None):
        embedding_output = self.emb_layer(
            input_ids=input_ids,
            position_ids=position_ids,
            token_type_ids=token_type_ids,
            inputs_embeds=inputs_embeds,
        )
        extended_attention_mask = self.bert.get_extended_attention_mask(
            attention_mask, input_ids.size(), "cpu")
        attn_outputs = self.attn_layer(
            embedding_output,
            attention_mask=extended_attention_mask,
            head_mask=head_mask,
            encoder_hidden_states=None,
            encoder_attention_mask=encoder_attention_mask,
            output_attentions=False
        )
        return attn_outputs[0]


bert_config = BertConfig.from_pretrained("bert-base-uncased")

target = "I like cats"
body = "Cats are nice to me"

tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

print("Target only with attention mask vector")
default_encoded = tokenizer(target, max_length=MAX_LENGTH,
                            padding="max_length", return_tensors="pt")
target_len = default_encoded["attention_mask"].sum().item()

seed_everything(0)
model = TestModel(bert_config)
def_outputs = model(**default_encoded)
attentions = def_outputs[0, :target_len, :].sum(1).tolist()
print(list(zip(default_encoded["input_ids"][0].tolist(), attentions)))
print()

print("Target only with attention mask matrix")
seed_everything(0)
model = TestModel(bert_config)
attn_matrix = torch.zeros((MAX_LENGTH, MAX_LENGTH), dtype=torch.long)
attn_matrix[:target_len, :target_len] = 1
default_encoded["attention_mask"] = attn_matrix
def_outputs = model(**default_encoded)
attentions = def_outputs[0, :target_len, :].sum(1).tolist()
print(list(zip(default_encoded["input_ids"][0].tolist(), attentions)))
print()


print("Target and body with directional attention")
config.load_yaml("configs/test.yaml")
config.Data.Encoder.max_seq_length.value = MAX_LENGTH
dir_encoder = DirectionalAttentionEncoder.from_config(config)
example = {"json": {"target": target, "body": body}}
dir_encoded = dir_encoder.encode_single_example(example)["json"]["encoded"]
dir_encoded = {k: v.unsqueeze(0) for (k, v) in dir_encoded.items()}
dir_encoded["token_type_ids"] = torch.zeros_like(dir_encoded["token_type_ids"])
input_len = (dir_encoded["input_ids"] != 0).int().sum().item()

seed_everything(0)
model = TestModel(bert_config)
dir_outputs = model(**dir_encoded)
attentions = dir_outputs[0, :input_len, :].sum(1).tolist()
print(list(zip(dir_encoded["input_ids"][0].tolist(), attentions)))
print()

print("Target and different body with directional attention")
config.load_yaml("configs/test.yaml")
config.Data.Encoder.max_seq_length.value = MAX_LENGTH
dir_encoder = DirectionalAttentionEncoder.from_config(config)
example = {"json": {"target": target, "body": "dogs are dumb"}}
dir_encoded = dir_encoder.encode_single_example(example)["json"]["encoded"]
dir_encoded = {k: v.unsqueeze(0) for (k, v) in dir_encoded.items()}
dir_encoded["token_type_ids"] = torch.zeros_like(dir_encoded["token_type_ids"])
input_len = (dir_encoded["input_ids"] != 0).int().sum().item()

seed_everything(0)
model = TestModel(bert_config)
dir_outputs = model(**dir_encoded)
attentions = dir_outputs[0, :input_len, :].sum(1).tolist()
print(list(zip(dir_encoded["input_ids"][0].tolist(), attentions)))

print("Target 
print(dir_encoded["attention_mask"])
