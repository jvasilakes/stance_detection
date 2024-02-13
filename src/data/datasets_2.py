import os
import json
import random
from copy import deepcopy
from glob import glob
from collections import defaultdict

import torch
import numpy as np
import webdataset as wds

#from .encoders import ENCODER_REGISTRY

torch.multiprocessing.set_sharing_strategy('file_system')


class DanishRumourDataset(object):

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
        encoder = None
        #if encoder_type is None:
        #    encoder = None
        #else:
        #    encoder = ENCODER_REGISTRY[encoder_type].from_config(config)
        return cls(datadir=config.Data.datadir.value,
                   encoder=encoder,
                   tasks_to_load=config.Data.tasks_to_load.value,
                   num_examples=config.Data.num_examples.value,
                   random_seed=config.Experiment.random_seed.value)

    def __init__(self,
                 datadir,
                 encoder=None,
                 tasks_to_load="all",
                 num_examples=-1,
                 random_seed=0,
                 example_format="pairs"):
        super().__init__()
        assert os.path.isdir(datadir), f"{datadir} is not a directory."
        self.datadir = datadir
        if encoder is None:
            self.encoder = lambda example: example
        else:
            self.encoder = encoder
        self.tasks_to_load = tasks_to_load
        self.num_examples = num_examples
        self.rng = random.Random(random_seed)
        self.example_format = example_format

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
        split_names = ["train", "val", "test"]
        split_files = [os.path.join(self.datadir, f"{split}.tar.gz")
                       for split in split_names]
        is_split = all([os.path.isfile(split_file)
                        for split_file in split_files])

        # splits will be three nested lists of [train, val, test]
        if is_split is True:
            splits = self.load_tar(self.datadir)
        else:
            all_examples = self.load_raw(self.datadir)
            splits = self.split(all_examples)
        return splits

    def load_tar(self, tardir):
        """
        Load the dataset already preprocessed and split
        into train, val, and test.
        """
        train_path = os.path.join(tardir, "train.tar.gz")
        val_path = os.path.join(tardir, "val.tar.gz")
        test_path = os.path.join(tardir, "test.tar.gz")
        train = wds.WebDataset(train_path).shuffle(
            1000, rng=self.rng).decode()
        train = train.map(self.encoder).map(self.tasks_filter)
        train = train.map(self.transform_labels)
        val = wds.WebDataset(val_path).decode()
        val = val.map(self.encoder).map(self.tasks_filter)
        val = val.map(self.transform_labels)
        test = wds.WebDataset(test_path).decode()
        test = test.map(self.encoder).map(self.tasks_filter)
        test = test.map(self.transform_labels)
        if self.num_examples > -1:
            # We have to call list, otherwise slice will return
            # different examples each epoch.
            train = list(train.slice(self.num_examples))
            val = list(val.slice(self.num_examples))
            test = list(test.slice(self.num_examples))
        return train, val, test

    def load_raw(self, datadir):
        all_examples = []
        for topic_dir in os.listdir(datadir):
            topic_path = os.path.join(datadir, topic_dir)
            thread_glob = os.path.join(topic_path, "*.json")
            for thread_file in glob(thread_glob):
                thread = json.load(open(thread_file))
                examples = self.get_examples(thread)
                all_examples.extend(examples)
        return all_examples

    def get_examples(self, thread):
        examples = []
        claim = thread["redditSubmission"]
        #if claim["IsRumour"] is False:
        #    return examples
        claim_id = claim["submission_id"]
        claim_text = claim["title"]
        if len(claim["text"]) > 0:
            claim_text = ': '.join([claim_text, claim["text"]])
        seen = set()
        for branch in thread["branches"]:
            for reply in branch:
                reply = reply["comment"]
                if reply["comment_id"] in seen:
                    continue
                seen.add(reply["comment_id"])

                reply_json = {"id": reply["submission_id"],
                              "text": reply["text"],
                              "labels": {"Stance": reply["SDQC_Submission"]}  # noqa
                              }
                if self.example_format == "stances_only":
                    example = {"reply": reply_json}
                    examples.append(example)
                elif self.example_format == "pairs":
                    claim_json = {"id": claim_id,
                                  "text": claim_text,
                                  "labels": {"Stance": claim["SourceSDQC"],
                                             "Rumour": claim["IsRumour"],
                                             "Veracity": claim["TruthStatus"]}
                                  }
                    example = {"claim": claim_json,
                               "reply": reply_json}
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

    def tasks_filter(self, sample):
        if self.tasks_to_load == "all":
            return sample
        sample["json"]["labels"] = {
            k: v for (k, v) in sample["json"]["labels"].items()
            if k in self.tasks_to_load}
        return sample

    def save(self, outdir):
        """
        Used for converting a dataset to webdataset tar format.
        """
        os.makedirs(outdir, exist_ok=False)
        train_path = os.path.join(outdir, "train.tar.gz")
        train_sink = wds.TarWriter(train_path, compress=True)

        val_path = os.path.join(outdir, "val.tar.gz")
        val_sink = wds.TarWriter(val_path, compress=True)

        test_path = os.path.join(outdir, "test.tar.gz")
        test_sink = wds.TarWriter(test_path, compress=True)

        splits = [self.train, self.val, self.test]
        sinks = [train_sink, val_sink, test_sink]
        for (split, sink) in zip(splits, sinks):
            for example in split:
                sink.write(self.inverse_transform_labels(example))
            sink.close()

    def summarize(self, by_split=True):
        if by_split is True:
            # split: task: label: count
            counts = defaultdict(lambda: defaultdict(lambda: defaultdict(int)))
        else:
            # task: label: count
            counts = defaultdict(lambda: defaultdict(int))
        for split in ["train", "val", "test"]:
            splitdata = getattr(self, split)
            for ex in splitdata:
                for (task, lab) in ex["reply"]["labels"].items():
                    if by_split is True:
                        counts[split][task][lab] += 1
                    else:
                        counts[task][lab] += 1

        for (key1, dict_i) in counts.items():
            print(key1)
            for (key2, dict_ii) in dict_i.items():
                if not isinstance(dict_ii, dict):
                    lab = key2
                    count = dict_ii
                    print(f"    {lab}: {count}")
                else:
                    print(" ", task)
                    for (lab, count) in dict_ii.items():
                        print(f"    {lab}: {count}")
