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

timestamp_pat = re.compile(r"<\|(\d+\.\d+)\|>")


def write_dataset_to_json(dataset, output_file_path, mode="w", encoding="utf-8", default=str, ensure_ascii=False):
    ds_iter = iter(dataset)
    with open(output_file_path, mode, encoding=encoding) as fo:
        for _, sample in enumerate(tqdm(ds_iter, desc="Writing to json", total=len(dataset), unit=" samples")):
            fo.write(f"{json.dumps(sample, default=default, ensure_ascii=ensure_ascii)}\n")


def normalize_text(s):
    s = s.strip('"')
    s = unicodedata.normalize("NFKD", s)  # normalize unicode chars
    s = re.sub(r"\[[^\]]*\]", "", s)  # remove content between square brackets
    s = re.sub(r"[´′’ʼ‘ʻ`]", "'", s)  # standardize quotes and apostrophes
    s = re.sub(r"[−‐–—]", "-", s)  # standardize hyphens and dashes
    s = re.sub(r"(?:…|\. \. \.)", "...", s)
    s = re.sub(r"\s+", " ", s).strip()  # replace any successive whitespace characters with a space
    return s


def maybe_round_up_timestamps(s):
    # round timestamp to nearest 0.02
    def _round_up(m):
        return f"<|{int(float(m) / 0.02) * 0.02:.2f}|>"

    return timestamp_pat.sub(lambda m: _round_up(m.group(1)), s)


def main(
    input_file_path: str,
    output_file_path: str,
    id_column_name: str = "id",
    # text_column_name: str = "text",
    whisper_transcript_column_name: str = "whisper_transcript",
    num_workers: int = 64,
    max_samples: Optional[int] = None,
):
    # load dataset
    dataset = load_dataset("json", data_files=input_file_path, split="train")
    print(dataset)

    # filter by source/target len ratio
    def process_function(example):
        # #letters
        # length_ratio = len(example[f"translated_{whisper_transcript_column_name}"]) / len(example[whisper_transcript_column_name])
        # #words
        length_ratio = len(example[f"translated_{whisper_transcript_column_name}"].split()) / len(example[whisper_transcript_column_name].split())
        # thr_ratio = 1.5
        thr_ratio = 1.3
        if length_ratio < 1 / thr_ratio or length_ratio > thr_ratio:
            print(length_ratio)
            # print(example["text"])
            # print(example["text_pnc"])
            print(example[whisper_transcript_column_name])
            print(example[f"translated_{whisper_transcript_column_name}"])
            print("\n\n")
            # sys.exit()
            return False

        # must surronded by timestamps
        s = example[f"translated_{whisper_transcript_column_name}"]
        start_match = re.search(r"^<\|(\d+\.\d+)\|>", s)
        end_match = re.search(r"<\|(\d+\.\d+)\|>$", s)
        if not (start_match and end_match):
            return False

        return True

    dataset = dataset.filter(
        process_function,
        num_proc=num_workers,
    )
    print(dataset)

    def process_function(example):
        example[whisper_transcript_column_name] = example[f"translated_{whisper_transcript_column_name}"]
        example[whisper_transcript_column_name] = maybe_round_up_timestamps(example[whisper_transcript_column_name])

        example["_task"] = "translate"
        example["_language"] = "en"

        return example

    dataset = dataset.map(
        process_function,
        remove_columns=[f"translated_{whisper_transcript_column_name}"],
        num_proc=num_workers,
    )

    whisper_transcript_mappings = dict(zip(dataset[id_column_name], dataset[whisper_transcript_column_name]))

    def process_function(example):

        # update prev
        def _decrement(num):
            l = len(num)
            return f"{int(num) - 1:0{l}d}"

        if example["condition_on_prev"]:
            prev_id = _decrement(example[id_column_name])
            if (prev_whisper_transcript := whisper_transcript_mappings.get(prev_id)):
                example[f"prev_{whisper_transcript_column_name}"] = prev_whisper_transcript
            else:
                # todo: false since already deleted and not translated
                example[f"prev_{whisper_transcript_column_name}"] = ""
                example["condition_on_prev"] = False

        return example

    dataset = dataset.map(
        process_function,
        # keep_in_memory=True,
        # load_from_cache_file=False,
        num_proc=num_workers,
    )


    # export
    write_dataset_to_json(dataset, output_file_path=output_file_path, mode="w")


if __name__ == "__main__":
    fire.Fire(main)
