#!/usr/bin/env python
# coding=utf-8
# Copyright 2024  Bofeng Huang

"""
python scripts/insert_column.py test_mozilla-foundation_common_voice_17_0_manifest.json test_mozilla-foundation_common_voice_17_0_manifest.json _language fr 64
"""


import json
from typing import Any

import fire
from datasets import load_dataset
from tqdm import tqdm


def write_dataset_to_json(dataset, output_file_path, mode="w", encoding="utf-8", default=str, ensure_ascii=False):
    ds_iter = iter(dataset)
    with open(output_file_path, mode, encoding=encoding) as fo:
        for _, sample in enumerate(tqdm(ds_iter, desc="Writing to json", total=len(dataset), unit=" samples")):
            fo.write(f"{json.dumps(sample, default=default, ensure_ascii=ensure_ascii)}\n")

    print(f"Saved manifest into {output_file_path}")


def main(
    input_file_path: str,
    output_file_path: str,
    column_name: str,
    column_value: Any,
    num_workers: int = 8,
):

    # load dataset
    dataset = load_dataset("json", data_files=input_file_path, split="train")

    dataset = dataset.map(lambda _: {column_name: column_value}, num_proc=num_workers)

    # export
    write_dataset_to_json(dataset, output_file_path=output_file_path, mode="w")


if __name__ == "__main__":
    fire.Fire(main)
