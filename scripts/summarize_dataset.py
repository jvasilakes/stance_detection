import os
import argparse

import pandas as pd


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("datadir", type=str)
    return parser.parse_args()


def main(args):
    bodies_file = os.path.join(args.datadir, "arc_bodies.csv")
    bodies = pd.read_csv(bodies_file)
    train_stances_file = os.path.join(args.datadir, "arc_stances_train.csv")
    train_stances = pd.read_csv(train_stances_file)
    test_stances_file = os.path.join(args.datadir, "arc_stances_test.csv")
    test_stances = pd.read_csv(test_stances_file)

    ntrain = len(train_stances)
    ntest = len(test_stances)

    train_stance_counts = pd.DataFrame(train_stances.Stance.value_counts())
    train_stance_counts['%'] = train_stance_counts / len(train_stances)

    test_stance_counts = pd.DataFrame(test_stances.Stance.value_counts())
    test_stance_counts['%'] = test_stance_counts / len(test_stances)

    stance_summary = pd.concat([train_stance_counts, test_stance_counts], axis=1)
    stance_summary.columns = ["train count", "train %", "test count", "test %"]
    print("STANCE")
    print("=======")
    print(stance_summary.to_markdown())

    summarize_columns(train_stances, test_stances, "Headline")
    summarize_columns(train_stances, test_stances, "Body ID")
    summarize_columns(train_stances, test_stances, ["Headline", "Body ID"])


def summarize_columns(train_df, test_df, columns):
    train_values = train_df[columns].values
    if len(train_values.shape) > 1:
        train_values = [tuple(x) for x in train_values]
    train = set(train_values)

    test_values = test_df[columns].values
    if len(test_values.shape) > 1:
        test_values = [tuple(x) for x in test_values]
    test = set(test_values)

    train_no_test = train.difference(test)
    test_no_train = test.difference(train)
    train_and_test = train.intersection(test)

    header = columns if isinstance(columns, str) else ', '.join(columns)
    print('\n' + header.upper())
    print("=======")
    print("Train")
    print(f"  N: {len(train)}")
    print(f"  - test: {len(train_no_test)}")
    print(f"  & test: {len(train_and_test)}")
    print("Test")
    print(f"  N: {len(test)}")
    print(f"  - train: {len(test_no_train)}")


if __name__ == "__main__":
    main(parse_args())
