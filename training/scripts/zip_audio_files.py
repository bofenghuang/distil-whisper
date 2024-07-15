#!/usr/bin/env python
# coding=utf-8
# Copyright 2024  Bofeng Huang

"""Pack audio files in manifest into zip file."""

import os
import sys

# Get the parent directory of the scripts directory
parent_dir = os.path.dirname(os.path.abspath(os.path.dirname(__file__)))

# Add the parent directory to the system path
sys.path.append(parent_dir)


import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import fire
import pandas as pd
from tqdm import tqdm
from utils.data_utils import create_zip, get_zip_manifest


def write_dataframe_to_json(df, output_file_path, mode="w", encoding="utf-8", default=str, ensure_ascii=False):
    df_records = df.to_dict("records")
    with open(output_file_path, mode, encoding=encoding) as fo:
        for _, sample in enumerate(tqdm(df_records, desc="Writing to json", unit=" samples")):
            fo.write(f"{json.dumps(sample, default=default, ensure_ascii=ensure_ascii)}\n")

    print(f"Saved manifest into {output_file_path}")


def pack_audio_files_into_zip(audio_dir):
    # print(f"Packing audio files in {audio_dir}")
    zip_filepath = f"{audio_dir}.zip"
    # Pack audios/features into zip
    create_zip(audio_dir, zip_filepath)
    # Fetch zip manifest
    # mapping of file_stem - new_zip_path
    audio_filepaths = get_zip_manifest(zip_filepath)
    return audio_filepaths


def main(
    input_manifest_filepath: str,
    output_manifest_filepath: str,
    preprocessing_num_workers: int = 8,
):
    # don't infer data type
    data_df = pd.read_json(input_manifest_filepath, lines=True, dtype=False)

    # debug
    # data_df = data_df.head(200)

    # sep dir and filename
    def _split_function(filepath):
        # return filepath.rsplit("/", 1)
        p = Path(filepath)
        return p.parent.as_posix(), p.stem

    data_df["audio_dir"], data_df["audio_filename"] = zip(*data_df["audio_filepath"].apply(_split_function))

    # sort by order in audio directories
    # data_df.sort_values(["audio_dir", "audio_filename"], inplace=True)

    # data_dfs_updated = []
    # for audio_dir, group_df in data_df.groupby("audio_dir"):
    #     print(f"Packing audio files in {audio_dir}")
    #     zip_filepath = f"{audio_dir}.zip"
    #     # Pack audios/features into zip
    #     create_zip(audio_dir, zip_filepath)
    #     # Fetch zip manifest
    #     audio_filepaths = get_zip_manifest(zip_filepath)
    #     # Update audio_filepath
    #     group_df["audio_zip_filepath"] = group_df["audio_filename"].map(audio_filepaths.get)
    #     data_dfs_updated.append(group_df)

    # # Concatenate all the group_dfs together
    # data_df = pd.concat(data_dfs_updated)

    with ProcessPoolExecutor(max_workers=preprocessing_num_workers) as executor:
        audio_filepaths = list(executor.map(pack_audio_files_into_zip, data_df["audio_dir"].unique().tolist()))

    # merge list of dicts
    audio_filepaths = {k: v for dct in audio_filepaths for k, v in dct.items()}

    # Update audio_filepath in DataFrame
    data_df["audio_zip_filepath"] = data_df["audio_filename"].map(audio_filepaths.get)

    # drop intermediate cols
    data_df.drop(["audio_dir", "audio_filename"], axis=1, inplace=True)

    write_dataframe_to_json(data_df, output_manifest_filepath)


if __name__ == "__main__":
    fire.Fire(main)
