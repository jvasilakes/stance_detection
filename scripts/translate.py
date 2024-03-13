import os
import json
import argparse
from glob import glob
from tqdm import tqdm
from copy import deepcopy
from itertools import zip_longest

import torch
from transformers import T5ForConditionalGeneration, T5Tokenizer


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("language", type=str, choices=["ru", "zh", "de", "pt"])
    parser.add_argument("datadir", type=str,
                        help="Directory containing {train,val,test}.jsonl")
    parser.add_argument("outdir", type=str)
    parser.add_argument("--device-num", "-D", type=int, default=0,
                        help="Device to use")
    parser.add_argument("--batch-size", "-B", type=int, default=128)
    return parser.parse_args()


def main(args):
    device = f"cuda:{args.device_num}"
    if args.language == "de":
        model_name = "aszfcxcgszdx/t5-large-en-de"
    elif args.language in ["zh", "ru"]:
        model_name = "utrobinmv/t5_translate_en_ru_zh_base_200"
    elif args.language == "pt":
        model_name = "unicamp-dl/translation-en-pt-t5"
    model = T5ForConditionalGeneration.from_pretrained(model_name)
    model = model.to(device)
    tokenizer = T5Tokenizer.from_pretrained(model_name)

    os.makedirs(args.outdir, exist_ok=False)

    fname_glob = os.path.join(args.datadir, "*.jsonl")
    for filepath in glob(fname_glob):
        fname = os.path.basename(filepath)
        examples = [json.loads(line) for line in open(filepath)]

        translated = []
        n_batches = len(examples) // args.batch_size
        for examples in tqdm(batch(examples, args.batch_size), desc=fname, total=n_batches):
            trans_examples = translate(examples, tokenizer, model, args.language, device=device)
            translated.extend(trans_examples)
        outpath = os.path.join(args.outdir, fname)
        with open(outpath, 'w') as outF:
            for ex in translated:
                json.dump(ex, outF)
                outF.write('\n')


def translate(examples, tokenizer, model, language, device="cpu"):
    prefix = f"translate to {language}: "
    trans_examples = deepcopy(examples)

    for text_type in ["target", "body"]:
        text_batch = [prefix + ex["json"][text_type] for ex in examples]
        inputs = tokenizer(text_batch, return_tensors="pt",
                           padding=True).to(device)
        generated_tokens = model.generate(**inputs)
        results = tokenizer.batch_decode(
                generated_tokens, skip_special_tokens=True)
        for (i, tex) in enumerate(trans_examples):
            tex["json"][text_type] = results[i]
        del inputs
    return trans_examples


def batch(iterable, batch_size=64):
    args = [iter(iterable)] * batch_size
    for batch in  zip_longest(*args, fillvalue=None):
        yield [ex for ex in batch if ex is not None]


if __name__ == "__main__":
    main(parse_args())
