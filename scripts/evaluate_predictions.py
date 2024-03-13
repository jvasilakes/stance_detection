import json
import argparse
import warnings

import numpy as np
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix


warnings.filterwarnings("ignore")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("predictions", type=str)
    return parser.parse_args()


def main(args):
    labels = []
    preds = []
    attention_vals = []
    for raw_data in open(args.predictions, 'r'):
        data = json.loads(raw_data)
        labels.append(data["json"]["label"])
        preds.append(data["json"]["prediction"])
        if "token_mask" in data["json"].keys():
            body_start = data["json"]["tokens"].index("[SEP]")
            attention_vals.append(data["json"]["token_mask"][body_start+1:])
    if len(attention_vals) > 0:
        all_attention_vals = np.concatenate(attention_vals)
        coverage = (all_attention_vals > 0).mean()
        all_attention_vals_nonzero = all_attention_vals[all_attention_vals > 0]
        attn_mean = all_attention_vals_nonzero.mean()
        attn_sd = all_attention_vals_nonzero.std()
        attn_min = all_attention_vals_nonzero.min()
        attn_max = all_attention_vals_nonzero.max()
        attn_chunks = compute_chunks(attention_vals)
        print("Attention Weights")
        print(f"  Coverage: {coverage*100:.3f}%")
        print(f"  Distribution (mean, sd): {attn_mean:.3f} +/- {attn_sd:.3f}")
        print(f"  Distribution (min, max): {attn_min:.8f} - {attn_max:.3f}")
        print(f"  Contiguity (num, len): {attn_chunks[:,0].mean():.3f}, {attn_chunks[:, 1].mean():.3f}")  # noqa
        print()

    sorted_labels = sorted(set(labels))
    cm = confusion_matrix(labels, preds, labels=sorted_labels)
    print_confusion_matrix(cm)
    p, r, f, _ = precision_recall_fscore_support(labels, preds, average=None,
                                                 labels=sorted_labels)
    pm, rm, fm, _ = precision_recall_fscore_support(
            labels, preds, average="micro", labels=sorted_labels)

    p = np.concatenate([p, [p.mean(), pm]])
    r = np.concatenate([r, [r.mean(), rm]])
    f = np.concatenate([f, [f.mean(), fm]])

    print()
    print(sorted_labels + ["Macro", "Micro"])
    for (metric_name, vals) in zip(["P", "R", "F"], [p, r, f]):
        val_str = ' '.join(f"{val:.4f}" for val in vals)
        print(f"{metric_name}: {val_str}")


def compute_chunks(attention_vals):
    all_chunks = []
    for vals in attention_vals:
        nonzero_vals = np.array(vals) > 0
        chunks = np.split(nonzero_vals, np.where(np.diff(nonzero_vals))[0]+1)
        nonzero_chunks = [c for c in chunks if c.all()]
        num_chunks = len(nonzero_chunks)
        chunk_lens = [len(chunk) for chunk in nonzero_chunks]
        all_chunks.append([num_chunks, np.mean(chunk_lens)])
    return np.array(all_chunks)


def print_confusion_matrix(cm):
    num_labs = cm.shape[0]
    row_data = []
    max_count_len = 0
    for i in range(num_labs):
        count_strs = [str(c) for c in cm[i]]
        max_row_str_len = max([len(sc) for sc in count_strs])
        if max_row_str_len > max_count_len:
            max_count_len = max_row_str_len
        row_data.append(count_strs)

    header = f"T↓/P→  {(' '*max_count_len).join(str(i) for i in range(num_labs))}"  # noqa
    print("\u0332".join(header + "  "))
    for i in range(num_labs):
        row_strs = [f"{s:<{max_count_len}}" for s in row_data[i]]
        print(f"{i:<5}│ {' '.join(row_strs)}")


if __name__ == "__main__":
    args = parse_args()
    main(args)
