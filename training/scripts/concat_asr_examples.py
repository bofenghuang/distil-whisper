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


def write_dataset_to_json(dataset, output_file_path, mode="w", encoding="utf-8", default=str, ensure_ascii=False):
    ds_iter = iter(dataset)
    with open(output_file_path, mode, encoding=encoding) as fo:
        for _, sample in enumerate(tqdm(ds_iter, desc="Writing to json", total=len(dataset), unit=" samples")):
            fo.write(f"{json.dumps(sample, default=default, ensure_ascii=ensure_ascii)}\n")

    print(f"Saved manifest into {output_file_path}")


def main(
    input_file_path: str,
    output_file_path: str,
    min_duration: float = 0.1,
    max_duration: float = 30.0,
    text_column_name: str = "text",
    whisper_transcript_column_name: str = "whisper_transcript",
    preprocessing_batch_size: int = 1000,  # Using a larger batch size results in a greater portion of audio samples being packed to 30-seconds, at the expense of higher memory consumption
    preprocessing_num_workers: int = 8,
    max_samples: Optional[int] = None,
):
    is_mcv = "common_voice" in input_file_path
    is_mls = "multilingual_librispeech" in input_file_path
    is_voxpopuli = "voxpopuli" in input_file_path
    is_yodas = "yodas" in input_file_path
    is_mtedx = "multilingual-tedx" in input_file_path

    id_column_name = (
        "audio_filepath" if is_mcv else "id" if (is_mls or is_mtedx) else "audio_id" if is_voxpopuli else "utt_id" if is_yodas else None
    )
    speaker_column_name = "speaker_id"
    audio_column_name = "audio_filepath"
    duration_column_name = "duration"
    language_column_name = "_language"

    # load dataset
    dataset = load_dataset("json", data_files=input_file_path, split="train")
    _print_ds_info(dataset, duration_column_name)

    has_text = text_column_name in dataset.column_names
    has_whisper_transcript = whisper_transcript_column_name in dataset.column_names
    has_language = language_column_name in dataset.column_names

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

        def _maybe_increment_timestamps(s, current_timestamp):
            def _inc(m, current_timestamp):
                return f"<|{float(m) + current_timestamp:.2f}|>"

            # round timestamp to nearest 0.02
            current_timestamp = int(current_timestamp / 0.02) * 0.02
            # increment timestamps if exist
            res = timestamp_pat.sub(lambda m: _inc(m.group(1), current_timestamp), s)
            # add space if not starting with timestamps
            return res if res.startswith("<|") else " " + res

        def _concat_and_save_wav_files(input_files, speaker_name):
            output_dir = input_files[0]
            output_dir = output_dir.replace("/train/", "/train_concatenated/")
            # output_dir = output_dir.replace("/train/", "/train_concatenated_20/")
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
            # text_column_name: [examples[text_column_name][0]],
            audio_column_name: [[examples[audio_column_name][0]]],
            "condition_on_prev": [False],
            # f"prev_{text_column_name}": [""],
            "is_concatenated": [],
        }
        if has_text:
            merged_examples[text_column_name] = [examples[text_column_name][0]]
        if has_whisper_transcript:
            merged_examples[whisper_transcript_column_name] = [examples[whisper_transcript_column_name][0]]
        if has_language:
            merged_examples[language_column_name] = [examples[language_column_name][0]]

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
                # merged_examples[text_column_name][-1] += _maybe_increment_timestamps(
                #     example[text_column_name], merged_examples[duration_column_name][-1]
                # )
                if has_text:
                    merged_examples[text_column_name][-1] += " " + example[text_column_name]
                if has_whisper_transcript:
                    # update timestamps in transcriptions
                    merged_examples[whisper_transcript_column_name][-1] += _maybe_increment_timestamps(
                        example[whisper_transcript_column_name], merged_examples[duration_column_name][-1]
                    )
                merged_examples[duration_column_name][-1] += example[duration_column_name]
                merged_examples[audio_column_name][-1].append(example[audio_column_name])
            else:
                if has_text:
                    merged_examples[text_column_name].append(example[text_column_name])
                if has_whisper_transcript:
                    merged_examples[whisper_transcript_column_name].append(example[whisper_transcript_column_name])
                if has_language:
                    merged_examples[language_column_name].append(example[language_column_name])
                merged_examples[speaker_column_name].append(example[speaker_column_name])
                merged_examples[duration_column_name].append(example[duration_column_name])
                merged_examples[audio_column_name].append([example[audio_column_name]])
                merged_examples["condition_on_prev"].append(True if is_same_speaker else False)

        # add last concatenated text as prev text
        # todo: condition on all or last example
        if has_text:
            merged_examples[f"prev_{text_column_name}"] = [""]
            for idx in range(1, len(merged_examples["condition_on_prev"])):
                merged_examples[f"prev_{text_column_name}"].append(
                    merged_examples[text_column_name][idx - 1] if merged_examples["condition_on_prev"][idx] else ""
                )
        if has_whisper_transcript:
            merged_examples[f"prev_{whisper_transcript_column_name}"] = [""]
            for idx in range(1, len(merged_examples["condition_on_prev"])):
                merged_examples[f"prev_{whisper_transcript_column_name}"].append(
                    merged_examples[whisper_transcript_column_name][idx - 1] if merged_examples["condition_on_prev"][idx] else ""
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

    dataset = dataset.map(
        concatenate_examples,
        batched=True,
        batch_size=preprocessing_batch_size,
        keep_in_memory=True,
        load_from_cache_file=False,
        remove_columns=dataset.column_names,
        num_proc=preprocessing_num_workers,
        desc="concatenating...",
    )
    _print_ds_info(dataset, duration_column_name)

    # add id column to keep segment order
    dataset = dataset.map(
        lambda _, idx: {"id": f"{idx:09d}"},
        with_indices=True,
        num_proc=preprocessing_num_workers,
    )

    # export
    write_dataset_to_json(dataset, output_file_path=output_file_path, mode="w")


if __name__ == "__main__":
    fire.Fire(main)
