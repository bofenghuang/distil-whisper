#!/usr/bin/env python
# coding=utf-8
# Copyright 2023  Bofeng Huang

"""Verify if entries in manifest exist."""

import json
import os
import re
import sys
from pathlib import Path

import fire
import soundfile as sf
from datasets import load_dataset
from tqdm import tqdm


def process(input_file_path: str, num_workers: int = 64):
    dataset = load_dataset("json", data_files=input_file_path, split="train")
    # print(dataset[0])
    # print(dataset.num_rows)

    # change audio file path
    """
    dataset = dataset.map(
        lambda x: {"audio_filepath": re.sub(r"^/projects/bhuang", "/gpfsscratch/rech/cjc/commun", x["audio_filepath"])},
        num_proc=8,
    )

    output_file_path = "/gpfsssd/scratch/rech/cjc/commun/corpus/speech/nemo_manifests/final/2023-11-28"
    output_file_path = f"{output_file_path}/{Path(input_file_path).name}"
    os.makedirs(os.path.dirname(output_file_path), exist_ok=True)
    with open(output_file_path, "w") as manifest_f:
        for sample in tqdm(dataset, desc="Saving", total=dataset.num_rows, unit=" samples"):
            manifest_f.write(f"{json.dumps(sample, ensure_ascii=False)}\n")
    quit()
    """

    def process_function(example):
        # p = Path(example["audio_filepath"])
        p = Path(example["audio_zip_filepath"])
        # assert p.exists(), example
        if not p.exists():
            print(example)
            sys.exit(1)
            return False

        # audio_info = sf.info(example["audio_filepath"])
        # # assert abs(audio_info.duration - example["duration"]) < 0.01, example
        # if abs(audio_info.duration - example["duration"]) > 0.01:
        #     return False

        return True

    # for line in tqdm(dataset, total=dataset.num_rows):
    #     process_function(line)

    dataset = dataset.filter(process_function, num_proc=num_workers)
    # print(dataset.num_rows)


def main(input_dir: str, num_workers: int = 64):

    for p in Path(input_dir).rglob("*.json"):
        input_file_path = p.as_posix()
        print(input_file_path)
        process(input_file_path, num_workers=num_workers)


if __name__ == "__main__":
    fire.Fire(main)
