#!/usr/bin/env python
# coding=utf-8
# Copyright 2024  Bofeng Huang

"""Normalize whisper transcriptions."""

import json
import re
from typing import Optional

import fire
from datasets import load_dataset
from tqdm import tqdm

timestamp_pat = re.compile(r"<\|\d+\.\d+\|>")


def write_dataset_to_json(dataset, output_file_path, mode="w", encoding="utf-8", default=str, ensure_ascii=False):
    ds_iter = iter(dataset)
    with open(output_file_path, mode, encoding=encoding) as fo:
        for _, sample in enumerate(tqdm(ds_iter, desc="Writing to json", total=len(dataset), unit=" samples")):
            fo.write(f"{json.dumps(sample, default=default, ensure_ascii=ensure_ascii)}\n")

    print(f"Saved manifest into {output_file_path}")


def main(
    input_file_path: str,
    output_file_path: str,
    num_workers: int = 8,
    max_samples: Optional[int] = None,
):

    # load dataset
    dataset = load_dataset("json", data_files=input_file_path, split="train")
    # _print_ds_info(dataset)

    # debug
    if max_samples is not None:
        dataset = dataset.select(range(max_samples))

    def normalize_text(s):
        # filter duplicated timestamps in the end
        # exists in mls it
        m = timestamp_pat.findall(s)
        if len(m) > 2 and m[-2] == m[-1]:
            s = re.sub(re.escape(m[-1]) + "$", "", s)

        return s

    dataset = dataset.map(
        lambda x: {"whisper_transcript": normalize_text(x["whisper_transcript"])},
        keep_in_memory=True,
        load_from_cache_file=False,
        num_proc=num_workers,
        desc="normalizing",
    )
    # _print_ds_info(dataset)

    # export
    write_dataset_to_json(dataset, output_file_path=output_file_path, mode="w")


if __name__ == "__main__":
    fire.Fire(main)
