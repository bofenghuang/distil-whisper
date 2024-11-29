#!/usr/bin/env python
# coding=utf-8
# Copyright 2024  Bofeng Huang

"""Filter by full upper-case transcriptions, wer, etc."""

import fire
import numpy as np
import pandas as pd


def _print_ds_info(df, duration_column_name="duration"):
    print(f"#utterances: {df.shape[0]}")
    durations = df["duration"]
    print(
        f"Duration statistics: tot {durations.sum() / 3600:.2f} h, mean {durations.mean():.2f} s, min {durations.min():.2f} s, max {durations.max():.2f} s"
    )
    print()


def main(input_file_path: str):

    print(input_file_path, "\n")

    # load dataset
    df = pd.read_json(input_file_path, lines=True)
    print("Raw dataset")
    _print_ds_info(df)

    # line = "-".join(input_file_path.rsplit("/", 4)[-4: -2])
    # line += f',{df["duration"].sum() / 3600}'

    # wer_cutoffs = list(range(0, 35, 5))
    wer_cutoffs = [20, 10, 5, 0]
    for wer_cutoff in wer_cutoffs:
        df_ = df[df["wer"] <= wer_cutoff]
        print(f"wer_cutoff: {wer_cutoff}")
        _print_ds_info(df_)

    #     line += f',{df_["duration"].sum() / 3600}'

    # output_file = "/lustre/fswork/projects/rech/gkb/uvl55hq/distil-whisper/training/tmp_dur.csv"
    # with open(output_file, "a") as f:
    #     f.write(line + "\n")

if __name__ == "__main__":
    fire.Fire(main)
