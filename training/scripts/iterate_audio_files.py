#!/usr/bin/env python
# coding=utf-8
# Copyright 2024  Bofeng Huang

"""Verify if entries in manifest exist."""

import os
import json
import sys
from tqdm import tqdm

# Get the parent directory of the scripts directory
parent_dir = os.path.dirname(os.path.abspath(os.path.dirname(__file__)))

# Add the parent directory to the system path
sys.path.append(parent_dir)

import fire
from datasets import load_dataset

from utils.audio_utils import get_waveform_from_audio_or_stored_zip

def write_dataset_to_json(dataset, output_file_path, mode="w", encoding="utf-8", default=str, ensure_ascii=False):
    ds_iter = iter(dataset)
    with open(output_file_path, mode, encoding=encoding) as fo:
        for _, sample in enumerate(tqdm(ds_iter, desc="Writing to json", total=len(dataset), unit=" samples")):
            fo.write(f"{json.dumps(sample, default=default, ensure_ascii=ensure_ascii)}\n")

def main(
    input_file_path: str,
    audio_column_name: str = "audio_zip_filepath",
    preprocessing_num_workers: int = 1,
):
    # load dataset
    dataset = load_dataset("json", data_files=input_file_path, split="train")

    def process_function(example):
        try:
            get_waveform_from_audio_or_stored_zip(example[audio_column_name])
            return True
        except Exception as e:
            print(str(e))
            print(example)
            sys.exit()
            return False

    dataset = dataset.filter(
        process_function,
        num_proc=preprocessing_num_workers,
    )

    # write_dataset_to_json(dataset, output_file_path=input_file_path[:-5] + "_new.json", mode="w")


if __name__ == "__main__":
    fire.Fire(main)
