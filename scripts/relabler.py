import os
import json
import argparse
from copy import deepcopy


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("labelmap", type=str,
                        help="Path to a JSON file containing a label mapping.")
    parser.add_argument("datadir", type=str,
                        help="Directory containing {train,val,test}.jsonl")
    parser.add_argument("outdir", type=str,
                        help="Where to save the relabeled files.")
    return parser.parse_args()


def main(args):
    labelmap = json.load(open(args.labelmap))
    os.makedirs(args.outdir, exist_ok=False)
    for split in ["train", "val", "test"]:
        datafile = os.path.join(args.datadir, f"{split}.jsonl")
        mapped_examples = []
        num_mapped = 0
        with open(datafile, 'r') as inF:
            for line in inF:
                example = json.loads(line)
                (mapped, success) = map_labels(example, labelmap)
                num_mapped += success
                mapped_examples.append(mapped)
        print(f"{split:<5} | remapped {num_mapped}")
        outfile = os.path.join(args.outdir, f"{split}.jsonl")
        with open(outfile, 'w') as outF:
            for ex in mapped_examples:
                json.dump(ex, outF)
                outF.write('\n')
    labelmap_logfile = os.path.join(args.outdir, "labelmap.json")
    with open(labelmap_logfile, 'w') as outF:
        json.dump(labelmap, outF)


def map_labels(example, labelmap):
    excp = deepcopy(example)
    for (task, val) in excp["json"]["labels"].items():
        try:
            excp["json"]["labels"][task] = labelmap[val]
            return (excp, 1)
        except KeyError:
            return (example, 0)


if __name__ == "__main__":
    main(parse_args())
