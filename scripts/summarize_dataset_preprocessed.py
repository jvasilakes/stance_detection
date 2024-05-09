import os
import json
import argparse
from collections import defaultdict


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("datadir", type=str,
                        help="Directory containing {train,val,test}.jsonl")
    return parser.parse_args()


def main(args):
    for split in ["train", "val", "test"]:
        datafile = os.path.join(args.datadir, f"{split}.jsonl")
        summary = defaultdict(lambda: defaultdict(int))
        with open(datafile, 'r') as inF:
            for line in inF:
                example = json.loads(line)
                labels = example["json"]["labels"]
                for (task, lab) in labels.items():
                    summary[task][lab] += 1
        
        print(split.upper())
        print('-' * len(split))
        for (task, label_counts) in summary.items():
            print(task)
            for (lab, count) in label_counts.items():
                print(f"  {lab}: {count}")
        print()


if __name__ == "__main__":
    main(parse_args())
