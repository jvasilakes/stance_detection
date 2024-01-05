import os
import sys
import argparse

sys.path.insert(0, os.path.abspath(os.getcwd()))
from src.data.dataset import StanceDataset


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("datadir", type=str,
                        help="Directory containing raw, unprocessed data.")
    parser.add_argument("outdir", type=str,
                        help="Where to save the processed dataset")
    return parser.parse_args()


def main(args):
    # The StanceDataset class takes care of deduplication
    # and splitting into train, val, test
    ds = StanceDataset(datadir=args.datadir)
    ds.save(args.outdir)
    ds.summarize()


if __name__ == "__main__":
    main(parse_args())
