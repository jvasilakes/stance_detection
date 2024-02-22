import os
import json
import argparse
from glob import glob
from tqdm import tqdm
from copy import deepcopy

from transformers import T5ForConditionalGeneration, T5Tokenizer


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("language", type=str, choices=["ru", "zh", "de", "pt"])
    parser.add_argument("datadir", type=str,
                        help="Directory containing {train,val,test}.jsonl")
    parser.add_argument("outdir", type=str)
    return parser.parse_args()


def main(args):
    if args.language == "de":
        model_name = "aszfcxcgszdx/t5-large-en-de"
    elif args.language in ["zh", "ru"]:
        model_name = "utrobinmv/t5_translate_en_ru_zh_base_200"
    elif args.language == "pt":
        model_name = "unicamp-dl/translation-en-pt-t5"
    model = T5ForConditionalGeneration.from_pretrained(model_name)
    tokenizer = T5Tokenizer.from_pretrained(model_name)

    os.makedirs(args.outdir, exist_ok=False)

    fname_glob = os.path.join(args.datadir, "*.jsonl")
    for filepath in glob(fname_glob):
        fname = os.path.basename(filepath)
        examples = [json.loads(line) for line in open(filepath)]

        translated = []
        i = 0
        for example in tqdm(examples, desc=fname):
            if i >= 3:
                break
            i += 1
            trans_example = translate(example, tokenizer, model)
            translated.append(trans_example)
        outpath = os.path.join(args.outdir, fname)
        with open(outpath, 'w') as outF:
            for ex in translated:
                json.dump(ex, outF)
                outF.write('\n')


def translate(example, tokenizer, model):
    prefix = 'translate to ru: '
    trans_example = deepcopy(example)

    for text_type in ["target", "body"]:
        text = example["json"][text_type]
        src_text = prefix + text
        # translate English to Russian
        input_ids = tokenizer(src_text, return_tensors="pt")
        generated_tokens = model.generate(**input_ids)
        result = tokenizer.batch_decode(
                generated_tokens, skip_special_tokens=True)
        trans_example["json"][text_type] = result[0]
    return trans_example


if __name__ == "__main__":
    main(parse_args())
