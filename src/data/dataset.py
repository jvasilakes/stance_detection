import os
import re
import json
import warnings
from glob import glob
from copy import deepcopy
from collections import defaultdict

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from src.data.util import ENCODER_REGISTRY, register_dataset


class AbstractStanceDataset(object):

    LABEL_ENCODINGS = {}

    @property
    def INVERSE_LABEL_ENCODINGS(self):
        try:
            return getattr(self, "_inverse_label_encodings")
        except AttributeError:
            self._inverse_label_encodings = {
                task: {enc_label: str_label for (str_label, enc_label)
                       in self.LABEL_ENCODINGS[task].items()}
                for task in self.LABEL_ENCODINGS}
            return self._inverse_label_encodings

    @classmethod
    def from_config(cls, config):
        encoder_type = config.Data.Encoder.encoder_type.value
        if encoder_type is None:
            encoder = None
        else:
            encoder = ENCODER_REGISTRY[encoder_type].from_config(config)
        return cls(datadir=config.Data.datadir.value,
                   encoder=encoder,
                   encode_labels=config.Data.Encoder.encode_labels,
                   tasks_to_load=config.Data.tasks_to_load.value,
                   num_examples=config.Data.num_examples.value,
                   random_seed=config.Experiment.random_seed.value)

    def __init__(self,
                 datadir,
                 encoder=None,
                 encode_labels=True,
                 tasks_to_load="all",
                 num_examples=-1,
                 random_seed=0):
        assert os.path.isdir(datadir), f"{datadir} is not a directory."
        self.datadir = datadir
        self.encoder = encoder
        self.encode_labels = encode_labels
        self.num_examples = num_examples

        if isinstance(tasks_to_load, str):
            if tasks_to_load == "all":
                tasks_to_load = list(self.LABEL_ENCODINGS.keys())
            else:
                tasks_to_load = [tasks_to_load]
        assert isinstance(tasks_to_load, (list, tuple))
        tasks_set = set(tasks_to_load)
        valid_tasks_set = set(self.LABEL_ENCODINGS)
        unknown_tasks = tasks_set.difference(valid_tasks_set)
        assert len(unknown_tasks) == 0, f"Unknown tasks: '{unknown_tasks}'"
        self.tasks_to_load = tasks_to_load

        self.train, self.val, self.test = self.load()

    @property
    def label_spec(self):
        return {task: len(labs) for (task, labs)
                in self.LABEL_ENCODINGS.items()
                if task in self.tasks_to_load}

    def load(self):
        datafiles = set(os.listdir(self.datadir))
        preprocessed_files = {"train.jsonl", "val.jsonl", "test.jsonl"}
        if datafiles == preprocessed_files:
            train, val, test = self.load_preprocessed()
        else:
            train, val, test = self.load_raw()
        return train, val, test

    def load_preprocessed(self):
        splits = []
        for split in ["train", "val", "test"]:
            splitfile = os.path.join(self.datadir, f"{split}.jsonl")
            examples = []
            for (i, line) in enumerate(open(splitfile, 'r')):
                if self.num_examples > 0 and i > self.num_examples:
                    break
                example = json.loads(line.strip())
                example = self.tasks_filter(example)
                if self.encode_labels is True:
                    example = self.transform_labels(example)
                if self.encoder is not None:
                    example = self.encoder(example)
                examples.append(example)
            splits.append(examples)
        return splits  # (train, val, test)

    def load_raw(self):
        raise NotImplementedError()

    def transform_labels(self, example):
        excp = deepcopy(example)
        label_dict = excp["json"].pop("labels")
        excp["json"]["labels"] = {task: self.LABEL_ENCODINGS[task][val]
                                  for (task, val) in label_dict.items()}
        return excp

    def inverse_transform_labels(self, example):
        excp = deepcopy(example)
        label_dict = excp["json"].pop("labels")
        excp["json"]["labels"] = {
            task: self.INVERSE_LABEL_ENCODINGS[task][val]
            for (task, val) in label_dict.items()}
        return excp

    def tasks_filter(self, example):
        if self.tasks_to_load == "all":
            return example
        labels = example["json"]["labels"]
        filtered_labels = {}
        for (task, lab) in labels.items():
            if task in self.tasks_to_load:
                filtered_labels[task] = lab
        example["json"]["labels"] = filtered_labels
        return example

    def save(self, outdir):
        os.makedirs(outdir, exist_ok=False)
        for split in ["train", "val", "test"]:
            outfile = os.path.join(outdir, f"{split}.jsonl")
            examples = getattr(self, split)
            with open(outfile, 'w') as outF:
                for example in examples:
                    json.dump(example, outF)
                    outF.write('\n')

    def summarize(self):
        summary_df = pd.DataFrame()
        total_targets = set()
        total_bodies = set()
        total_stance_counts = defaultdict(int)
        for split in ["train", "val", "test"]:
            examples = getattr(self, split)
            targets = set()
            bodies = set()
            stance_counts = defaultdict(int)
            for example in examples:
                targets.add(example["json"]["target"])
                total_targets.add(example["json"]["target"])
                bodies.add(example["json"]["body"])
                total_bodies.add(example["json"]["body"])
                stance = example["json"]["labels"]["Stance"]
                stance_counts[stance] += 1
                total_stance_counts[stance] += 1
            stance_counts_percs = {
                stance: f"{count} ({100*(count/len(examples)):.0f}%)"
                for (stance, count) in stance_counts.items()}
            summ = pd.DataFrame({"N": len(examples),
                                 "Targets": len(targets),
                                 "Bodies": len(bodies),
                                 **dict(stance_counts_percs)},
                                index=[split])
            summary_df = pd.concat([summary_df, summ])
        total = pd.DataFrame({"N": len(self.train) + len(self.val) + len(self.test),  # noqa
                              "Targets": len(total_targets),
                              "Bodies": len(total_bodies),
                              **dict(total_stance_counts)},
                             index=["Total"])
        summary_df = pd.concat([summary_df, total])
        print(summary_df.to_markdown())


@register_dataset("arc")
class ARCStanceDataset(AbstractStanceDataset):

    LABEL_ENCODINGS = {
        "Stance": {"disagree": 0,
                   "agree": 1,
                   "discuss": 2,
                   "unrelated": 3}
    }

    def load_raw(self):
        """
        example = {"__key__": str  # unique ID for this example
                   "__url__": str  # the source file for this example
                   "json": {
                            "target": str  # the target text
                            "body": str    # the body text of the response
                            "labels": {task: label}  # for all tasks
                   }
                  }
        """
        warnings.warn("Data is not encoded! You should save this data split with dm.save(outdir) and then load it again to use it as input to a model.")  # noqa
        bodies_file = os.path.join(self.datadir, "bodies.csv")
        bodies = pd.read_csv(bodies_file, index_col=0)  # index by body id
        stances_train_file = os.path.join(
            self.datadir, "stances_train.csv")
        train_stances = pd.read_csv(stances_train_file)
        stances_test_file = os.path.join(self.datadir, "stances_test.csv")
        test_stances = pd.read_csv(stances_test_file)

        # Remove duplicates
        train_stances = train_stances[~train_stances.duplicated()]
        test_stances = test_stances[~test_stances.duplicated()]

        # Remove from train the examples that are in both
        merged = pd.merge(train_stances, test_stances,
                          on=["Headline", "Body ID"], how="inner")
        train_ = train_stances._append(merged)
        train_["duplicated"] = train_.duplicated(["Headline", "Body ID"], keep=False)  # noqa
        train_stances = train_[~train_["duplicated"]][["Headline", "Body ID", "Stance"]]  # noqa

        train_stances["Body"] = bodies.loc[train_stances["Body ID"]].articleBody.values  # noqa
        test_stances["Body"] = bodies.loc[test_stances["Body ID"]].articleBody.values  # noqa

        def get_example(row, stance_file=''):
            return {"__key__": row["Body ID"],
                    "__url__": stance_file,
                    "json": {"target": row["Headline"],
                             "body": row["Body"],
                             "labels": {"Stance": row["Stance"]}
                             }
                    }

        train_idxs, val_idxs = train_test_split(train_stances.index,
                                                stratify=train_stances.Stance,
                                                test_size=0.2)
        tmp_train = train_stances.loc[train_idxs]
        val_stances = train_stances.loc[val_idxs]
        train_stances = tmp_train

        train = train_stances.apply(
            get_example, stance_file=stances_train_file, axis=1).tolist()
        val = val_stances.apply(
            get_example, stance_file=stances_train_file, axis=1).tolist()
        test = test_stances.apply(
            get_example, stance_file=stances_test_file, axis=1).tolist()
        if self.num_examples > 0:
            train = train[:self.num_examples]
            val = val[:self.num_examples]
            test = test[:self.num_examples]
        return train, val, test


@register_dataset("rumoureval")
class RumourEvalTaskADataset(AbstractStanceDataset):

    LABEL_ENCODINGS = {
        "Stance": {"deny": 0,
                   "support": 1,
                   "query": 2,
                   "comment": 3},
        "Veracity": {"false": 0,
                     "true": 1,
                     "unverified": 2}
    }

    @classmethod
    def from_config(cls, config):
        encoder_type = config.Data.Encoder.encoder_type.value
        encoder = None
        if encoder_type is None:
            encoder = None
        else:
            encoder = ENCODER_REGISTRY[encoder_type].from_config(config)
        return cls(datadir=config.Data.datadir.value,
                   encoder=encoder,
                   encode_labels=config.Data.Encoder.encode_labels,
                   tasks_to_load=config.Data.tasks_to_load.value,
                   num_examples=config.Data.num_examples.value,
                   random_seed=config.Experiment.random_seed.value,
                   **config.Data.dataset_kwargs.value)

    def __init__(self,
                 datadir,
                 encoder=None,
                 encode_labels=True,
                 tasks_to_load="all",
                 num_examples=-1,
                 random_seed=0,
                 version=2019,
                 load_reddit=False,
                 example_format="pairs"):
        assert version in [2017, 2019]
        if version == 2017:
            assert load_reddit is False, "2017 has no reddit data."
        self.version = version
        self.load_reddit = load_reddit
        assert example_format in ["stances_only", "pairs", "conversations"]
        self.example_format = example_format
        super().__init__(datadir, encoder=encoder, encode_labels=encode_labels,
                         tasks_to_load=tasks_to_load, num_examples=num_examples,
                         random_seed=random_seed)

    def load_raw(self):
        warnings.warn("Data is not encoded! You should save this data split with dm.save(outdir) and then load it again to use it as input to a model.")  # noqa

        if self.version == 2019:
            train_dir = os.path.join(self.datadir, "rumoureval-2019-training-data")  # noqa
            test_dir = os.path.join(self.datadir, "rumoureval-2019-test-data")

            twitter_train_dir = os.path.join(train_dir, "twitter-english")
            twitter_test_dir = os.path.join(test_dir, "twitter-en-test-data")

            train_label_file = open(os.path.join(train_dir, "train-key.json"))
            val_label_file = open(os.path.join(train_dir, "dev-key.json"))
            test_label_file = open(os.path.join(
                self.datadir, "final-eval-key.json"))
            train_labels = json.load(train_label_file)
            val_labels = json.load(val_label_file)
            test_labels = json.load(test_label_file)

        elif self.version == 2017:
            train_dir = os.path.join(self.datadir, "semeval2017-task8-dataset")

            twitter_train_dir = os.path.join(train_dir, "rumoureval-data")
            twitter_test_dir = os.path.join(
                    self.datadir, "semeval2017-task8-test-data")

            train_label_file = open(os.path.join(
                train_dir, "traindev", "rumoureval-subtaskA-train.json"))
            val_label_file = open(os.path.join(
                train_dir, "traindev", "rumoureval-subtaskA-dev.json"))
            test_label_file = open(os.path.join(
                self.datadir, "test_taska.json"))

            train_labels = json.load(train_label_file)
            train_labels = {"subtaskaenglish": train_labels}
            val_labels = json.load(val_label_file)
            val_labels = {"subtaskaenglish": val_labels}
            test_labels = json.load(test_label_file)
            test_labels = {"subtaskaenglish": test_labels}

        all_train = self.get_examples_twitter(
            twitter_train_dir, train_labels)
        all_val = self.get_examples_twitter(
            twitter_train_dir, val_labels)
        all_test = self.get_examples_twitter(
            twitter_test_dir, test_labels)

        # self.print_twitter(
        #     twitter_train_dir, train_labels)

        if self.load_reddit is True:
            reddit_train_dir = os.path.join(train_dir, "reddit-training-data")
            reddit_val_dir = os.path.join(train_dir, "reddit-dev-data")
            reddit_test_dir = os.path.join(test_dir, "reddit-test-data")

            reddit_train = self.get_examples_reddit(
                reddit_train_dir, train_labels)
            all_train = all_train + reddit_train

            reddit_val = self.get_examples_reddit(reddit_val_dir, val_labels)
            all_val = all_val + reddit_val

            reddit_test = self.get_examples_reddit(
                reddit_test_dir, test_labels)
            all_test = all_test + reddit_test

        return all_train, all_val, all_test

    def get_examples_twitter(self, twitter_dir, labels):
        all_examples = []
        for topic in os.listdir(twitter_dir):
            topic_dir = os.path.join(twitter_dir, topic)
            for thread in os.listdir(topic_dir):
                thread_dir = os.path.join(topic_dir, thread)
                examples = self.get_examples(
                    thread_dir, labels, kind="twitter")
                all_examples.extend(examples)
        return all_examples

    def get_examples_reddit(self, reddit_dir, labels):
        all_examples = []
        for thread in os.listdir(reddit_dir):
            thread_dir = os.path.join(reddit_dir, thread)
            examples = self.get_examples(thread_dir, labels, kind="reddit")
            all_examples.extend(examples)
        return all_examples

    def get_examples(self, thread_dir, labels, kind="twitter"):
        tree_structure = json.load(open(os.path.join(
            thread_dir, "structure.json")))

        src_file = os.listdir(os.path.join(thread_dir, "source-tweet"))[0]
        src_path = os.path.join(thread_dir, "source-tweet", src_file)

        claim = json.load(open(src_path))
        claim_id = os.path.splitext(src_file)[0]
        posts = {claim_id: claim}
        reply_dir = os.path.join(thread_dir, "replies")
        for fname in os.listdir(reply_dir):
            reply = json.load(open(os.path.join(reply_dir, fname)))
            reply_id = os.path.splitext(fname)[0]
            posts[reply_id] = reply

        return self.get_examples_from_tree(
            tree_structure, posts, labels, kind=kind, url=thread_dir)

    def get_examples_from_tree(self, tree_structure, posts, labels,
                               headline=None, kind="twitter", url=''):
        """
        example = {"__key__": str  # unique ID for this example
                   "__url__": str  # the source file for this example
                   "json": {
                            "target": str  # the target text
                            "body": str    # the body text of the response
                            "labels": {task: label}  # for all tasks
                   }
                  }
        """
        task_2_subtask = {"Stance": "subtaskaenglish",
                          "Veracity": "subtaskbenglish"}

        examples = []
        if tree_structure == []:
            return examples
        for (key, subtree) in tree_structure.items():
            try:
                if kind == "twitter":
                    post = self.get_tweet(posts, key)
                elif kind == "reddit":
                    post = self.get_reddit_post(posts, key)
            except KeyError:
                continue
            example_labels = {}
            for task in self.tasks_to_load:
                subtask = task_2_subtask[task]
                try:
                    lab = labels[subtask][key]
                    example_labels[task] = lab
                except KeyError:
                    # Keep whatever labels we find.
                    pass
            if example_labels == {}:
                continue

            if headline is None:
                headline = post

            new_examples = self.get_examples_from_tree(
                subtree, posts, labels, headline=headline, kind=kind, url=url)

            key = '_'.join([str(headline["id"]), str(post["id"])])
            example = {"__key__": key,
                       "__url__": url,
                       "json": {
                                "target": '',
                                "body": '',
                                "labels": {}
                       }
                       }
            example["json"]["body"] = post["text"]
            example["json"]["labels"] = example_labels
            if self.example_format == "stances_only":
                examples.append(example)
                examples.extend(new_examples)
            elif self.example_format == "pairs":
                example["json"]["target"] = headline["text"]
                examples.append(example)
                examples.extend(new_examples)
            elif self.example_format == "conversations":
                raise NotImplementedError("example_format = conversations")
                if headline is None:
                    example = {"claim": post,
                               "label": example_labels}
                else:
                    example = {"reply": post,
                               "label": example_labels}
                if len(new_examples) > 0:
                    example = [example]
                    example.append(new_examples)
                examples.append(example)
        return examples

    def get_tweet(self, posts, key):
        tweet = posts[key]
        tweet = {"id": tweet["id"],
                 "text": tweet["text"],
                 "source": "twitter"}
        return tweet

    def get_reddit_post(self, posts, key):
        post = posts[key]
        post = post["data"]
        if "children" in post.keys():
            if isinstance(post["children"][0], str):
                raise KeyError()
            post = post["children"][0]["data"]
            text = post["title"]
        else:
            text = post["body"]
        post = {"id": post["id"],
                "text": text,
                "source": "reddit"}
        return post

    def print_twitter(self, twitter_dir, labels):
        for topic in os.listdir(twitter_dir):
            topic_dir = os.path.join(twitter_dir, topic)
            for thread in os.listdir(topic_dir):
                thread_dir = os.path.join(topic_dir, thread)
                self.print_thread_tree(thread_dir, labels)
                input()

    def print_thread_tree(self, thread_dir, labels):
        tree_structure = json.load(open(os.path.join(
            thread_dir, "structure.json")))

        src_file = os.listdir(os.path.join(thread_dir, "source-tweet"))[0]
        src_path = os.path.join(thread_dir, "source-tweet", src_file)

        claim = json.load(open(src_path))
        claim_id = os.path.splitext(src_file)[0]
        tweets = {claim_id: claim}
        reply_dir = os.path.join(thread_dir, "replies")
        for fname in os.listdir(reply_dir):
            reply = json.load(open(os.path.join(reply_dir, fname)))
            reply_id = os.path.splitext(fname)[0]
            tweets[reply_id] = reply

        self.__print_thread_tree(tree_structure, tweets, labels)

    def __print_thread_tree(self, tree_structure, tweets, labels, indent=0):
        task_2_subtask = {"Stance": "subtaskaenglish",
                          "Veracity": "subtaskbenglish"}

        if tree_structure == []:
            return
        for (key, subtree) in tree_structure.items():
            try:
                tweet = tweets[key]["text"]
            except KeyError:
                continue
            try:
                example_labels = {}
                for task in self.tasks_to_load:
                    subtask = task_2_subtask[task]
                    example_labels[task] = labels[subtask][key]
            except KeyError:
                # Keep whatever labels we have.
                pass
            if "comment" in example_labels.values():
                continue
            label_str = f"{[l for l in example_labels.values()]}"
            print(' ' * indent + tweet + f"  *** {label_str} ***")
            self.__print_thread_tree(subtree, tweets, labels, indent=indent+4)


@register_dataset("danish")
class DanishRumourDataset(AbstractStanceDataset):

    LABEL_ENCODINGS = {
            "Stance": {"Supporting": 0,
                       "Denying": 1,
                       "Querying": 2,
                       "Commenting": 3},
            "Veracity": {"False": 0,
                         "True": 1,
                         "Unverified": 2,
                         "Underspeci": 3},
            "Rumour": {"False": 0,
                       "True": 1}
    }

    @classmethod
    def from_config(cls, config):
        encoder_type = config.Data.Encoder.encoder_type.value
        encoder = None
        if encoder_type is None:
            encoder = None
        else:
            encoder = ENCODER_REGISTRY[encoder_type].from_config(config)
        return cls(datadir=config.Data.datadir.value,
                   encoder=encoder,
                   encode_labels=config.Data.Encoder.encode_labels,
                   tasks_to_load=config.Data.tasks_to_load.value,
                   num_examples=config.Data.num_examples.value,
                   random_seed=config.Experiment.random_seed.value,
                   **config.Data.dataset_kwargs.value)

    def __init__(self,
                 datadir,
                 encoder=None,
                 encode_labels=True,
                 tasks_to_load="all",
                 num_examples=-1,
                 random_seed=0,
                 example_format="pairs"):
        self.example_format = example_format
        super().__init__(datadir, encoder=encoder, encode_labels=encode_labels,
                         tasks_to_load=tasks_to_load, num_examples=num_examples,
                         random_seed=random_seed)

    def load_raw(self):
        """
        example = {"__key__": str  # unique ID for this example
                   "__url__": str  # the source file for this example
                   "json": {
                            "target": str  # the target text
                            "body": str    # the body text of the response
                            "labels": {task: label}  # for all tasks
                   }
                  }
        """
        warnings.warn("Data is not encoded! You should save this data split with dm.save(outdir) and then load it again to use it as input to a model.")  # noqa
        all_examples = []
        for topic_dir in os.listdir(self.datadir):
            topic_path = os.path.join(self.datadir, topic_dir)
            thread_glob = os.path.join(topic_path, "*.json")
            for thread_file in glob(thread_glob):
                examples = self.get_examples(thread_file)
                all_examples.extend(examples)
        return self.split(all_examples)

    def get_examples(self, thread_file):
        thread = json.load(open(thread_file))
        examples = []
        claim = thread["redditSubmission"]
        claim_id = claim["submission_id"]
        claim_text = claim["title"]
        if len(claim["text"]) > 0:
            claim_text = ': '.join([claim_text, claim["text"]])
        seen = set()
        for branch in thread["branches"]:
            for reply in branch:
                reply = reply["comment"]
                reply_id = reply["comment_id"]
                if reply_id in seen:
                    continue
                seen.add(reply_id)
                key = '_'.join([claim_id, reply_id])
                example = {"__key__": key,
                           "__url__": thread_file,
                           "json": {
                               "target": '',
                               "body": '',
                               "labels": {}
                           }
                           }
                example["json"]["body"] = reply["text"]
                example["json"]["labels"]["Stance"] = reply["SDQC_Submission"]  # noqa
                if self.example_format == "stances_only":
                    examples.append(example)
                elif self.example_format == "pairs":
                    example["json"]["target"] = claim_text
                    example["json"]["labels"]["Rumour"] = str(claim["IsRumour"])  # noqa
                    example["json"]["labels"]["Veracity"] = str(claim["TruthStatus"])  # noqa
                    examples.append(example)
                elif self.example_format == "conversation":
                    raise NotImplementedError("example_format = conversation")

        return examples

    def split(self, examples):
        train = []
        val = []
        test = []
        splits = [train, val, test]
        for (i, example) in enumerate(examples):
            split_idx = np.random.choice(range(3), p=[0.8, 0.1, 0.1])
            splits[split_idx].append(example)
        return splits


@register_dataset("rustance")
class RussianStanceDataset(AbstractStanceDataset):

    LABEL_ENCODINGS = {
            "Stance": {"s": 0,
                       "d": 1,
                       "q": 2,
                       "c": 3},
    }

    def load_raw(self):
        """
        example = {"__key__": str  # unique ID for this example
                   "__url__": str  # the source file for this example
                   "json": {
                            "target": str  # the target text
                            "body": str    # the body text of the response
                            "labels": {task: label}  # for all tasks
                   }
                  }
        """
        warnings.warn("Data is not encoded! You should save this data split with dm.save(outdir) and then load it again to use it as input to a model.")  # noqa
        datafile = os.path.join(self.datadir, "dataset.csv")
        df = pd.read_csv(datafile, delimiter=';').reset_index()
        examples = list(df.apply(
            self.get_example, axis=1, source_file=datafile))
        return self.split(examples)

    def get_example(self, df_row, source_file):
        example = {"__key__": df_row["index"],
                   "__url__": source_file,
                   "json": {"target": df_row["Title"],
                            "body": df_row["Text"],
                            "labels": {"Stance": df_row["Stance"]}
                            }
                   }
        return example

    def split(self, examples):
        train = []
        val = []
        test = []
        splits = [train, val, test]
        for (i, example) in enumerate(examples):
            split_idx = np.random.choice(range(3), p=[0.8, 0.1, 0.1])
            splits[split_idx].append(example)
        return splits


@register_dataset("arastance")
class AraStanceDataset(AbstractStanceDataset):

    LABEL_ENCODINGS = {
            "Stance": {"Agree": 0,
                       "Disagree": 1,
                       "Discuss": 2,
                       "Unrelated": 3}
    }

    def load_raw(self):
        warnings.warn("Data is not encoded! You should save this data split with dm.save(outdir) and then load it again to use it as input to a model.")  # noqa
        splits = []
        for split in ["train", "dev", "test"]:
            splitfile = os.path.join(self.datadir, f"{split}.jsonl")
            splits.append(self.load_file(splitfile))
        return splits  # [train, val, test]

    def load_file(self, splitfile):
        """
        example = {"__key__": str  # unique ID for this example
                   "__url__": str  # the source file for this example
                   "json": {
                            "target": str  # the target text
                            "body": str    # the body text of the response
                            "labels": {task: label}  # for all tasks
                   }
                  }
        """
        examples = []
        data = [json.loads(line) for line in open(splitfile)]
        for datum in data:
            for (i, article) in enumerate(datum["article"]):
                example = {"__key__": datum["filename"],
                           "__url__": datum["claim_url"],
                           "json": {
                                    "target": datum["claim"],
                                    "body": article,
                                    "labels": {"Stance": datum["stance"][i]}
                                    }
                           }
                examples.append(example)
        return examples


@register_dataset("imdb")
class IMDBDataset(AbstractStanceDataset):

    LABEL_ENCODINGS = {
                "Stance": {"negative": 0,
                           "positive": 1}
    }

    def load_raw(self):
        splits = []
        for split in ["train", "test"]:
            examples = self.load_split(os.path.join(self.datadir, split))
            if split == "train":
                labels = [ex["json"]["labels"]["Stance"] for ex in examples]
                train, val = train_test_split(examples, test_size=2000, stratify=labels)
                splits.append(train)
                splits.append(val)
            else:
                splits.append(examples)
        return splits

    def load_split(self, splitdir):
        examples = []

        posdir = os.path.join(splitdir, "pos")
        posfiles = glob(os.path.join(posdir, "*.txt"))
        examples.extend(self._get_examples(posfiles, "positive"))

        negdir = os.path.join(splitdir, "neg")
        negfiles = glob(os.path.join(negdir, "*.txt"))
        examples.extend(self._get_examples(negfiles, "negative"))

        return examples

    def _get_examples(self, files, label):
        REPLACE_NO_SPACE = re.compile("[.;:!\'?,\"()\[\]]")
        REPLACE_WITH_SPACE = re.compile("(<br\s*/><br\s*/>)|(\-)|(\/)")
        for f in files:
            with open(f, 'r') as inF:
                text = inF.read()

            text = text.strip()
            text = REPLACE_NO_SPACE.sub("", text)
            text = REPLACE_WITH_SPACE.sub("", text)
            
            example = {"__key__": os.path.basename(os.path.splitext(f)[0]),
                       "__url__": f,
                       "json": {"target": '',
                                "body": text,
                                "labels": {"Stance": label}}
                       }
            yield example

