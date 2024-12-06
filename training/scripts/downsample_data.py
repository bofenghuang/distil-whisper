#!/usr/bin/env python
# coding=utf-8
# Copyright 2024  Bofeng Huang

"""Downsample dataset to required total duration."""

import json
import os
import random, re

import fire
import numpy as np
from datasets import load_dataset
from tqdm import tqdm


def _print_ds_info(ds, duration_column_name="duration"):
    print()
    print(f"#rows: {ds.num_rows}")
    print(f"Columns: {ds.column_names}")
    # ds_df = ds.to_pandas()
    durations = np.asarray(ds[duration_column_name])
    print(
        f"Duration statistics: tot {durations.sum() / 3600:.2f}h, mean {durations.mean():.2f}s, median {np.median(durations):.2f}s, min {durations.min():.2f}s, max {durations.max():.2f}s"
    )
    print()


def write_dataset_to_json(dataset, output_file_path, mode="w", encoding="utf-8", default=str, ensure_ascii=False):
    ds_iter = iter(dataset)
    with open(output_file_path, mode, encoding=encoding) as fo:
        for _, sample in enumerate(tqdm(ds_iter, desc="Writing to json", total=len(dataset), unit=" samples")):
            fo.write(f"{json.dumps(sample, default=default, ensure_ascii=ensure_ascii)}\n")

    print(f"Saved manifest into {output_file_path}")


def main(
    train_files: str,
    output_file_path: str,
    max_duration: int = 5_000,
):
    ext = train_files.rsplit(".", 1)[-1]
    train_files = train_files.split("+")
    train_files = train_files if len(train_files) > 1 else train_files[0]
    dataset = load_dataset(ext, data_files=train_files, split="train")
    _print_ds_info(dataset)

    dataset = dataset.filter(lambda x: x["wer"] <= 10, num_proc=32)
    _print_ds_info(dataset)

    dataset = dataset.filter(lambda x: x["duration"] >= 20, num_proc=32)
    _print_ds_info(dataset)

    # init or will take long time
    durations = dataset["duration"]
    indexes = random.sample(range(dataset.num_rows), dataset.num_rows)

    cur_dur = 0
    for i, idx in enumerate(tqdm(indexes)):
        if cur_dur + durations[idx] > max_duration * 3_600:
            break
        cur_dur += durations[idx]

    indexes = indexes[:i]

    dataset = dataset.select(indexes)
    _print_ds_info(dataset)

    # tmp
    # dataset = dataset.rename_columns(
    #     {
    #         "_language": "lang",
    #         "text": "original_text",
    #         "text_pnc": "text",
    #     }
    # )

    # def norm_(s):
    #     s = re.sub((r"<\|(\d+\.\d+)\|>"), "", s)
    #     s = re.sub(r"\s+", " ", s).strip()  # replace any successive whitespace characters with a space
    #     return s

    # dataset = dataset.map(
    #     lambda example: {"text": norm_(example["whisper_transcript"])},
    #     num_proc=32,
    # )

    os.makedirs(os.path.dirname(output_file_path), exist_ok=True)
    write_dataset_to_json(dataset, output_file_path=output_file_path, mode="w")

    # dur dist by lang
    # datadf = dataset.to_pandas()
    # print(datadf.groupby("_language")["duration"].sum() / 3600)


if __name__ == "__main__":
    fire.Fire(main)
