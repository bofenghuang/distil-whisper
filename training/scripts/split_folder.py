#!/usr/bin/env python
# coding=utf-8
# Copyright 2023  Bofeng Huang

import json
import os
import shutil
from math import ceil
from pathlib import Path

import fire
from datasets import load_dataset
from tqdm import tqdm


def write_dataset_to_json(dataset, output_file_path, mode="w", encoding="utf-8", default=str, ensure_ascii=False):
    ds_iter = iter(dataset)
    with open(output_file_path, mode, encoding=encoding) as fo:
        for _, sample in enumerate(tqdm(ds_iter, desc="Writing to json", total=len(dataset), unit=" samples")):
            fo.write(f"{json.dumps(sample, default=default, ensure_ascii=ensure_ascii)}\n")

    print(f"Saved manifest into {output_file_path}")


def generate_folder_mapping(source_dir, max_files=5_000):
    source_path = Path(source_dir)

    # files = [f for f in source_path.iterdir() if f.is_file()]
    files = [f for f in source_path.iterdir() if f.is_file() and f.suffix == ".wav"]
    total_files = len(files)

    num_subfolders = ceil(total_files / max_files)
    # print(f"Folder {source_folder} ({total_files} files) will be split into {num_subfolders} subfolders")

    mapping = {}
    for i in range(num_subfolders):
        start_idx = i * max_files
        end_idx = min((i + 1) * max_files, total_files)

        for file_path in files[start_idx:end_idx]:
            source = str(file_path)
            target = str(file_path.parent / f"{i:08d}" / file_path.name)
            mapping[source] = target

    return mapping


def main(input_file_path, output_file_path, num_workers=8):

    dataset = load_dataset("json", data_files=input_file_path, split="train")
    print(dataset)

    source_path = Path(input_file_path).parent
    source_dir = str(source_path)

    # source_dir = "/projects/bhuang/corpus/speech/nemo_manifests/facebook/multilingual_librispeech/french/train_concatenated"

    # source_path = Path(source_dir)
    source_subdir_paths = [f for f in source_path.iterdir() if f.is_dir()]
    # print(source_subdir_paths)

    if not source_subdir_paths:
        file_mapping = generate_folder_mapping(source_dir)
    else:
        file_mapping = {}
        for source_subdir_path in source_subdir_paths:
            # print(source_subdir_path)
            file_mapping.update(generate_folder_mapping(str(source_subdir_path)))

    print(f"#mappings: {len(file_mapping)}")

    assert len(file_mapping) == dataset.num_rows
    # print(file_mapping)

    def process_function(example):
        source = example["audio_filepath"]
        target = file_mapping[source]
        os.makedirs(os.path.dirname(target), exist_ok=True)
        shutil.move(source, target)

        example["audio_filepath"] = target
        return example

    dataset = dataset.map(process_function, num_proc=num_workers, desc="moving files...")

    write_dataset_to_json(dataset, output_file_path=output_file_path, mode="w")


if __name__ == "__main__":
    fire.Fire(main)
