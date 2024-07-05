#!/usr/bin/env python
# coding=utf-8
# Copyright 2024  Bofeng Huang

"""Filter by full upper-case transcriptions, wer, etc."""

import json
import re
from typing import Optional

import fire
import numpy as np
from datasets import load_dataset
from tqdm import tqdm

timestamp_pat = re.compile(r"<\|\d+\.\d+\|>")


def write_dataset_to_json(dataset, output_file_path, mode="w", encoding="utf-8", default=str, ensure_ascii=False):
    ds_iter = iter(dataset)
    with open(output_file_path, mode, encoding=encoding) as fo:
        for _, sample in enumerate(tqdm(ds_iter, desc="Writing to json", total=len(dataset), unit=" samples")):
            fo.write(f"{json.dumps(sample, default=default, ensure_ascii=ensure_ascii)}\n")

    print(f"Saved manifest into {output_file_path}")


def _print_ds_info(ds, duration_column_name="duration"):
    print()
    print(f"#rows: {ds.num_rows}")
    print(f"Columns: {ds.column_names}")
    durations = np.asarray(ds[duration_column_name])
    print(
        f"Duration statistics: tot {durations.sum() / 3600:.2f}h, mean {durations.mean():.2f}s, min {durations.min():.2f}s, max {durations.max():.2f}s"
    )
    print()


def main(
    input_file_path: str,
    output_file_path: str,
    wer_threshold: Optional[int] = None,
    min_duration: Optional[int] = None,
    num_workers: int = 8,
    max_samples: Optional[int] = None,
):

    # load dataset
    dataset = load_dataset("json", data_files=input_file_path, split="train")
    _print_ds_info(dataset)

    # debug
    if max_samples is not None:
        dataset = dataset.select(range(max_samples))

    def filter_function(example):

        # not empty reference (norm)
        if not example["text_norm"]:
            return False

        # filter entirely upper-case transcriptions: these are erroneous generations from large-v3
        # this usually happens in short segments of jargon, name, etc in mcv
        if (
            example["whisper_transcript"] is not None
            and example["whisper_transcript"].upper() == example["whisper_transcript"]
        ):
            return False

        # or leave this into an option in training script
        if wer_threshold is not None and example["wer"] > wer_threshold:
            return False

        if min_duration is not None and example["duration"] < min_duration:
            return False

        # todo: audio len, text len

        return True

    dataset = dataset.filter(
        filter_function,
        # lambda *x, **y: not filter_function(*x, **y),  # debug
        keep_in_memory=True,
        load_from_cache_file=False,
        num_proc=num_workers,
        desc="filtering",
    )
    _print_ds_info(dataset)

    # export
    write_dataset_to_json(dataset, output_file_path=output_file_path, mode="w")


if __name__ == "__main__":
    fire.Fire(main)
