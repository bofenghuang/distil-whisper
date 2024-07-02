#!/usr/bin/env python
# coding=utf-8
# Copyright 2024  Bofeng Huang

"""Verify if entries in manifest exist."""

import hashlib
import json
import os
import re
import wave
from typing import Optional

import fire
import numpy as np
from datasets import load_dataset
from datasets.arrow_dataset import table_iter
from tqdm import tqdm


timestamp_pat = re.compile(r"<\|(\d+\.\d+)\|>")


def md5(to_hash: str, encoding: str = "utf-8") -> str:
    try:
        return hashlib.md5(to_hash.encode(encoding), usedforsecurity=False).hexdigest()
    except TypeError:
        return hashlib.md5(to_hash.encode(encoding)).hexdigest()  # nosec


def concat_wav_files(input_files, output_file):
    with wave.open(output_file, "wb") as wav_out:
        for input_file in input_files:
            with wave.open(input_file, "rb") as wav_in:
                if not wav_out.getnframes():
                    wav_out.setparams(wav_in.getparams())
                wav_out.writeframes(wav_in.readframes(wav_in.getnframes()))


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


def write_dataset_to_json(dataset, output_file_path, mode="w", default=str, ensure_ascii=False):
    ds_iter = iter(dataset)
    with open(output_file_path, mode) as fo:
        for _, sample in enumerate(tqdm(ds_iter, desc="Writing to json", total=len(dataset), unit=" samples")):
            fo.write(f"{json.dumps(sample, default=default, ensure_ascii=ensure_ascii)}\n")

    print(f"Saved manifest into {output_file_path}")


def main(
    input_file_path: str,
    output_file_path: str,
    min_duration: float = 0.1,
    max_duration: float = 30.0,
    preprocessing_batch_size: int = 1000,  # Using a larger batch size results in a greater portion of audio samples being packed to 30-seconds, at the expense of higher memory consumption
    preprocessing_num_workers: int = 8,
    max_samples: Optional[int] = None,
):
    is_mcv = "common_voice" in input_file_path
    is_mls = "multilingual_librispeech" in input_file_path
    is_voxpopuli = "voxpopuli" in input_file_path
    is_yodas = "yodas" in input_file_path

    id_column_name = (
        "audio_filepath" if is_mcv else "id" if is_mls else "audio_id" if is_voxpopuli else "utt_id" if is_yodas else None
    )
    speaker_column_name = "speaker_id"
    duration_column_name = "duration"
    audio_column_name = "audio_filepath"
    text_column_name = "text"

    # load dataset
    dataset = load_dataset("json", data_files=input_file_path, split="train")
    _print_ds_info(dataset, duration_column_name)

    # filter out >30s
    dataset = dataset.filter(
        # ~0s segment can be just badly segmented but text exists in its neibour segments' audio
        # lambda x: min_duration <= x[duration_column_name] and x[duration_column_name] <= max_duration,
        lambda x: x[duration_column_name] <= max_duration,
        num_proc=preprocessing_num_workers,
        desc="filtering by duration...",
    )
    _print_ds_info(dataset, duration_column_name)

    # preprocess
    if is_mcv:
        dataset = dataset.map(
            lambda x: {speaker_column_name: x["client_id"]},
            num_proc=preprocessing_num_workers,
            desc="preprocessing...",
        )
    if is_mls:
        dataset = dataset.map(
            lambda x: {speaker_column_name: str(x[speaker_column_name]) + "-" + str(x["chapter_id"])},
            num_proc=preprocessing_num_workers,
            desc="preprocessing...",
        )
    if is_yodas:
        dataset = dataset.map(
            lambda x: {speaker_column_name: x["utt_id"].lstrip("-").split("-", 1)[0]},
            num_proc=preprocessing_num_workers,
            desc="preprocessing...",
        )

    # sort
    dataset = dataset.sort(id_column_name)

    # debug
    if max_samples is not None:
        dataset = dataset.select(range(max_samples))

    def concatenate_examples(examples):
        # print(examples)

        def _increment_timestamps(s, current_timestamp):
            def _inc(m, current_timestamp):
                return f"<|{float(m) + current_timestamp:.2f}|>"

            return timestamp_pat.sub(lambda m: _inc(m.group(1), current_timestamp), s)

        def _concat_and_save_wav_files(input_files, speaker_name):
            output_dir = input_files[0]
            output_dir = output_dir.replace("/train/", "/train_concatenated/")
            output_dir = output_dir.rsplit("/", 1)[0]
            output_file_name = md5("+".join([x.rsplit("/", 1)[1].rsplit(".", 1)[0] for x in input_files]))
            output_file_name = speaker_name + "-" + output_file_name + ".wav"
            output_file = output_dir + "/" + output_file_name
            os.makedirs(output_dir, exist_ok=True)
            concat_wav_files(input_files, output_file)
            return output_file

        merged_examples = {
            speaker_column_name: [examples[speaker_column_name][0]],
            duration_column_name: [examples[duration_column_name][0]],
            text_column_name: [examples[text_column_name][0]],
            audio_column_name: [[examples[audio_column_name][0]]],
            "condition_on_prev": [False],
            # "prev_text": [""],
            "is_concatenated": [],
        }

        # for example_id, example in enumerate(examples):
        for example_id, example in enumerate(table_iter(examples.pa_table, batch_size=1)):
            # print(example)
            example = examples.formatter.format_row(example)

            if example_id == 0:
                continue

            is_same_speaker = merged_examples[speaker_column_name][-1] == example[speaker_column_name]
            is_concatenable = (merged_examples[duration_column_name][-1] + example[duration_column_name]) <= max_duration

            if is_same_speaker and is_concatenable:
                # inplace concatenation
                # merged_examples[text_column_name][-1] += " " + example[text_column_name]
                # update timestamps in transcriptions
                merged_examples[text_column_name][-1] += " " + _increment_timestamps(
                    example[text_column_name], merged_examples[duration_column_name][-1]
                )
                merged_examples[duration_column_name][-1] += example[duration_column_name]
                merged_examples[audio_column_name][-1].append(example[audio_column_name])
            else:
                merged_examples[text_column_name].append(example[text_column_name])
                merged_examples[speaker_column_name].append(example[speaker_column_name])
                merged_examples[duration_column_name].append(example[duration_column_name])
                merged_examples[audio_column_name].append([example[audio_column_name]])
                merged_examples["condition_on_prev"].append(True if is_same_speaker else False)

        # add last concatenated text as prev text
        # todo: condition on all or last example
        merged_examples["prev_text"] = [""]
        for idx in range(1, len(merged_examples["condition_on_prev"])):
            merged_examples["prev_text"].append(
                merged_examples[text_column_name][idx - 1] if merged_examples["condition_on_prev"][idx] else ""
            )

        # concat audios
        for idx in range(len(merged_examples[audio_column_name])):
            # save flag for if is concatenated
            merged_examples["is_concatenated"].append(True if len(merged_examples[audio_column_name][idx]) > 1 else False)
            # concat and save
            merged_examples[audio_column_name][idx] = _concat_and_save_wav_files(
                merged_examples[audio_column_name][idx], merged_examples[speaker_column_name][idx]
            )

        return merged_examples

    processed_dataset = dataset.map(
        concatenate_examples,
        batched=True,
        batch_size=preprocessing_batch_size,
        num_proc=preprocessing_num_workers,
        remove_columns=dataset.column_names,
        load_from_cache_file=False,
        desc="concatenating...",
    )
    _print_ds_info(processed_dataset, duration_column_name)

    # export
    write_dataset_to_json(processed_dataset, output_file_path=output_file_path, mode="w")


if __name__ == "__main__":
    fire.Fire(main)
