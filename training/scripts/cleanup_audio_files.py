#!/usr/bin/env python
# coding=utf-8
# Copyright 2024  Bofeng Huang

import os
from concurrent.futures import ThreadPoolExecutor

import fire
from datasets import load_dataset
from tqdm import tqdm


def main(
    input_file_a: str,
    input_file_b: str,
    audio_column_name: str = "audio_filepath",
    num_workers: int = 64,
):
    dataset_a = load_dataset("json", data_files=input_file_a, split="train")
    print(f"Loaded {dataset_a.num_rows:,d} examples")

    dataset_b = load_dataset("json", data_files=input_file_b, split="train")
    print(f"Loaded {dataset_b.num_rows:,d} examples")

    files_a = set(dataset_a[audio_column_name])
    files_b = set(dataset_b[audio_column_name])
    files_to_delete = list(files_a - files_b)
    print(f"Found {len(files_to_delete):,d} files to delete")

    def delete_file(file_path):
        try:
            if os.path.exists(file_path):
                os.remove(file_path)
            else:
                print(f"Didn't find {file_path}")
        except Exception as e:
            print(f"Failed to delete {file_path} with error {str(e)}")

    # Process in batches to manage memory
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        results = list(tqdm(executor.map(delete_file, files_to_delete), total=len(files_to_delete), desc="Deleting files"))


if __name__ == "__main__":
    fire.Fire(main)
