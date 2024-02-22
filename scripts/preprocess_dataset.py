import os
import sys
import argparse

sys.path.insert(0, os.path.abspath(os.getcwd()))
from src.data import DATASET_REGISTRY  # noqa


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset_name", type=str,
                        help="The registered name of the dataset.")
    parser.add_argument("datadir", type=str,
                        help="""Path to directory containing
                                dataset to preprocess.""")
    parser.add_argument("outdir", type=str,
                        help="Where to save the processed dataset.")
    return parser.parse_args()


def main(args):
    dataset_cls = DATASET_REGISTRY[args.dataset_name]
    ds = dataset_cls(args.datadir)
    ds.save(args.outdir)
    ds.summarize()


if __name__ == "__main__":
    main(parse_args())
