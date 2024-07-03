#!/usr/bin/env python
# coding=utf-8
# Copyright 2024  Bofeng Huang


import json
import re
from typing import Optional

import fire
import numpy as np
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
    id_column_name: str = "id",
    # text_column_name: str = "text",
    whisper_transcript_column_name: str = "whisper_transcript",
    num_workers: int = 64,
    max_samples: Optional[int] = None,
):
    # load dataset
    dataset = load_dataset("json", data_files=input_file_path, split="train")
    print(dataset)

    # id_2_whisper_transcript_mappings = dict(zip(dataset[id_column_name], dataset[whisper_transcript_column_name]))

    # dataset = dataset.map(
    #     lambda x: {
    #         f"prev_{whisper_transcript_column_name}": (
    #             id_2_whisper_transcript_mappings[x[id_column_name] - 1] if x["condition_on_prev"] else ""
    #         )
    #     },
    #     # keep_in_memory=True,
    #     load_from_cache_file=False,
    #     num_proc=num_workers,
    # )

    # sort
    dataset = dataset.sort(id_column_name)

    # debug
    if max_samples is not None:
        dataset = dataset.select(range(max_samples))

    dataset = dataset.map(
        lambda x, idx: {
            f"prev_{whisper_transcript_column_name}": (
                dataset[whisper_transcript_column_name][idx - 1] if x["condition_on_prev"] else ""
            )
        },
        # keep_in_memory=True,
        load_from_cache_file=False,
        num_proc=num_workers,
    )

    # export
    write_dataset_to_json(dataset, output_file_path=output_file_path, mode="w")


if __name__ == "__main__":
    fire.Fire(main)
