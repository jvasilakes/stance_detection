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
    parser.add_argument(
        "-k", "--kwargs", nargs=2, metavar=("GROUP.PARAM", "VALUE"),
        action="append", default=[],
        help="""Keyword argument to pass to the dataset __init__
        E.g., -k year 2017 -k load_reddit True""")
    return parser.parse_args()


def maybe_cast(val):
    """
    Command line keyword arguments are always str.
    Try to cast them to the relevant type.
    """
    if val.lower() == "true":
        return True
    elif val.lower() == "false":
        return False
    else:
        try:
            return int(val)
            return float(val)
        except ValueError:
            return val


def main(args):
    dataset_kwargs = {k: maybe_cast(v) for (k, v) in args.kwargs}
    dataset_cls = DATASET_REGISTRY[args.dataset_name]
    ds = dataset_cls(args.datadir, **dataset_kwargs)
    ds.save(args.outdir)
    ds.summarize()


if __name__ == "__main__":
    main(parse_args())
