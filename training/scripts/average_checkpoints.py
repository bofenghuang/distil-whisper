#!/usr/bin/env python
# coding=utf-8
# Copyright 2024  Bofeng Huang

"""
Average last N checkpoints.
Adapted from: https://github.com/espnet/espnet/blob/master/utils/average_checkpoints.py
"""

import re
import shutil
from pathlib import Path

import fire
from safetensors.torch import load_file, safe_open, save_file
from tqdm import tqdm


def copy_dir_with_exclusions(src_dir, dst_dir, exclude_patterns=None):
    if exclude_patterns is None:
        exclude_patterns = []

    regex_patterns = [re.compile(pattern) for pattern in exclude_patterns]

    def should_exclude(path):
        for pattern in regex_patterns:
            if pattern.search(str(path)):
                return True
        return False

    src_path = Path(src_dir)
    dst_path = Path(dst_dir)

    dst_path.mkdir(exist_ok=True)

    for item in src_path.rglob("*"):
        # Get relative path
        relative_path = item.relative_to(src_path)
        destination = dst_path / relative_path

        # Skip if path matches exclusion pattern
        if should_exclude(relative_path):
            # print(f"Excluding: {relative_path}")
            continue

        if item.is_dir():
            destination.mkdir(exist_ok=True)
        else:
            shutil.copy2(item, destination)
            print(f"Copied: {relative_path}")


def get_last_n_checkpoints(input_dir, n):
    checkpoints = []

    for p in Path(input_dir).iterdir():
        if p.is_dir() and p.name.startswith("checkpoint-"):
            try:
                # Extract the number after 'checkpoint-'
                num = int(p.name.split("-")[1])
                checkpoints.append((num, p))
            except (IndexError, ValueError):
                continue

    checkpoints.sort(reverse=True)
    return [(p / "model.safetensors").as_posix() for _, p in checkpoints[:n]]


def main(
    input_dir: str,
    n: int = 5,
):

    model_files = get_last_n_checkpoints(input_dir, n=n)

    average = None
    metadata = None

    # sum
    for model_file in tqdm(model_files):
        model = load_file(model_file)
        if average is None:
            average = model

            with safe_open(model_file, framework="pt") as f:
                metadata = f.metadata()
        else:
            for k in average.keys():
                # only average decoder params
                if "decoder" in k:
                    average[k] += model[k]

    # average
    for k in average.keys():
        if average[k] is not None:
            # only average decoder params
            if "decoder" in k:
                average[k] /= n

    # new_dir = f"{input_dir}_avg{n}"

    # copy
    # copy_dir_with_exclusions(input_dir, new_dir, exclude_patterns=[r"checkpoint-\d+-epoch-\d+", r"model\.safetensors"])

    # save
    # save_file(average, f"{new_dir}/model.safetensors")
    # save_file(average, f"{new_dir}/model.safetensors", metadata=metadata)

    save_file(average, f"{input_dir}/model_avg{n}.safetensors", metadata=metadata)


if __name__ == "__main__":
    fire.Fire(main)
