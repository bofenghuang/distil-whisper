#!/usr/bin/env python
# coding=utf-8
# Copyright 2024  Bofeng Huang

"""Filter by len ratio, update prev, add lang/task, check timestamps"""


import json
import re
import sys
import unicodedata
from typing import Optional

import fire
import numpy as np
import pandas as pd
from datasets import Dataset, load_dataset
from tqdm import tqdm


def write_dataset_to_json(dataset, output_file_path, mode="w", encoding="utf-8", default=str, ensure_ascii=False):
    ds_iter = iter(dataset)
    with open(output_file_path, mode, encoding=encoding) as fo:
        for _, sample in enumerate(tqdm(ds_iter, desc="Writing to json", total=len(dataset), unit=" samples")):
            fo.write(f"{json.dumps(sample, default=default, ensure_ascii=ensure_ascii)}\n")


def main(
    input_file_path: str,
    output_file_path: str,
    # id_column_name: str = "id",
    text_column_name: str = "text",
    num_workers: int = 1,
    max_samples: Optional[int] = None,
):
    # load dataset
    dataset = load_dataset("json", data_files=input_file_path, split="train")
    print(dataset)

    # filter by source/target len ratio
    def process_function(example):
        def _norm(s):
            s = s.strip('"')
            # s = unicodedata.normalize("NFKD", s)  # normalize unicode chars
            s = re.sub(r"\[[^\]]*\]", "", s)  # remove content between square brackets
            # s = re.sub(r"[´′’ʼ‘ʻ`]", "'", s)  # standardize quotes and apostrophes
            # s = re.sub(r"[−‐–—]", "-", s)  # standardize hyphens and dashes
            # s = re.sub(r"(?:…|\. \. \.)", "...", s)
            s = re.sub(r"\s+", " ", s).strip()  # replace any successive whitespace characters with a space
            return s

        # #letters
        # length_ratio = len(example[f"{text_column_name}_pnc"]) / len(example[text_column_name])
        # #words
        # length_ratio = len(example[f"{text_column_name}_pnc"].split()) / len(example[text_column_name].split())
        norm_text = _norm(example[text_column_name])
        if not norm_text or not example[f"{text_column_name}_pnc"]:
            return False
        length_ratio = len(example[f"{text_column_name}_pnc"].split()) / len(norm_text.split())
        # thr_ratio = 1.5
        thr_ratio = 1.3
        if length_ratio < 1 / thr_ratio or length_ratio > thr_ratio:
            print(length_ratio)
            # print(example["text"])
            # print(example["text_pnc"])
            print(example[text_column_name])
            print(example[f"{text_column_name}_pnc"])
            print("\n\n")
            # sys.exit()
            return False

        return True

    dataset = dataset.filter(
        process_function,
        num_proc=num_workers,
    )
    print(dataset)

    # export
    write_dataset_to_json(dataset, output_file_path=output_file_path, mode="w")


if __name__ == "__main__":
    fire.Fire(main)
