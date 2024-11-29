#!/usr/bin/env python
# coding=utf-8
# Copyright 2024  Bofeng Huang

"""
Synthesize code-switching multilingual segments.

Adapted from:
https://developer.nvidia.com/blog/multilingual-and-code-switched-automatic-speech-recognition-with-nvidia-nemo/#how_to_train_a_multilingual_model
https://github.com/NVIDIA/NeMo/tree/main/scripts/speech_recognition/code_switching
"""

import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import hashlib
import json
import random
import re
from pathlib import Path
from typing import Union

import fire
import numpy as np

# from scipy.io import wavfile
import soundfile as sf
from datasets import Dataset, load_dataset
from joblib import Parallel, delayed
from tqdm import tqdm

from utils.audio_utils import get_waveform_from_audio_or_stored_zip, get_waveform

SAMPLE_RATE = 16_000
timestamp_pat = re.compile(r"<\|(\d+\.\d+)\|>")


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


def md5(to_hash: str, encoding: str = "utf-8") -> str:
    try:
        return hashlib.md5(to_hash.encode(encoding), usedforsecurity=False).hexdigest()
    except TypeError:
        return hashlib.md5(to_hash.encode(encoding)).hexdigest()  # nosec


def concat_examples(
    examples,
    output_audio_dir: str,
    max_pause_beg_s: Union[int, float],
    min_pause_join_s: Union[int, float],
    max_pause_join_s: Union[int, float],
    max_pause_end_s: Union[int, float],
    audio_amplitude_normalization: int,
    speaker_column_name: str = "speaker_id",
):
    def _maybe_increment_timestamps(s, current_timestamp):
        def _inc(m, current_timestamp):
            return f"<|{float(m) + current_timestamp:.2f}|>"

        # round timestamp to nearest 0.02
        current_timestamp = int(current_timestamp / 0.02) * 0.02
        # increment timestamps if exist
        res = timestamp_pat.sub(lambda m: _inc(m.group(1), current_timestamp), s)
        # add space if not starting with timestamps
        return res if res.startswith("<|") else " " + res

    combined_audio = []
    output_filename = ""
    last_lang_token = ""
    merged_example = {
        "text": "",
        "whisper_transcript": "",
        "duration": 0,
        "speaker_id": "",
        "_language": "auto",  # todo
        "condition_on_prev": False,
        "is_concatenated": len(examples) > 1,
        "prev_text": "",
        "prev_whisper_transcript": "",
    }

    pause_beg_s = random.uniform(0, max_pause_beg_s)
    # pause_join_s = random.uniform(0, max_pause_join_s)
    pause_join_s = random.uniform(min_pause_join_s, max_pause_join_s)
    pause_end_s = random.uniform(0, max_pause_end_s)

    staring_pause = np.zeros(int(pause_beg_s * SAMPLE_RATE))
    combined_audio += list(staring_pause)
    merged_example["duration"] += pause_beg_s

    for index, example in enumerate(examples):
        data_sample, _ = get_waveform_from_audio_or_stored_zip(example["audio_filepath"])
        # data_sample, _ = get_waveform(example["audio_filepath"], always_2d=False, output_sample_rate=SAMPLE_RATE, normalize_volume=True)

        # didn't trim to keep whisper timestamps
        # Remove leading and trailing zeros
        # data_sample = np.trim_zeros(data_sample)

        # segment withs all sample equal to 0
        if np.maximum(np.abs(data_sample.max()), np.abs(data_sample.min())) < 1e-5:
            # print(example)
            # sys.exit(1)
            continue

        # normalizing data
        data_sample_norm = (
            data_sample / np.maximum(np.abs(data_sample.max()), np.abs(data_sample.min())) * audio_amplitude_normalization
        )

        combined_audio += list(data_sample_norm)

        # output_filename += Path(example["audio_filepath"]).stem
        output_filename += example["audio_filepath"]
        merged_example["text"] += example["text"]

        # only insert lang token when different from last segment
        if (lang_token := f'<|{example["_language"]}|>') != last_lang_token:
            merged_example["whisper_transcript"] += lang_token
        last_lang_token = lang_token

        merged_example["whisper_transcript"] += _maybe_increment_timestamps(
            example["whisper_transcript"], merged_example["duration"]
        )
        merged_example["speaker_id"] += example[speaker_column_name]
        merged_example["duration"] += example["duration"]

        # adding small pause between semgments
        if index != (len(examples) - 1):
            pause = np.zeros(int(pause_join_s * SAMPLE_RATE))
            combined_audio += list(pause)

            output_filename += "+"
            merged_example["text"] += " "
            merged_example["speaker_id"] += "-"
            merged_example["duration"] += pause_join_s

    ending_pause = np.zeros(int(pause_end_s * SAMPLE_RATE))
    combined_audio += list(ending_pause)
    merged_example["duration"] += pause_end_s

    # os.makedirs(output_audio_dir, exist_ok=True)
    # audio_file_path = output_audio_dir + "/" + md5(output_filename) + ".wav"
    # random dump into 2000 subfolders, to not exceed upload limit
    # todo: define #folders
    output_audio_sub_dir = f"{output_audio_dir}/{random.choice(range(1000)):08d}"
    os.makedirs(output_audio_sub_dir, exist_ok=True)
    audio_file_path = output_audio_sub_dir + "/" + md5(output_filename) + ".wav"

    # wavfile.write(audio_file_path, fs, np.array(combined_audio).astype(np.int16))
    # Alternative-  librosa.output.write_wav(audio_file_path, combined_audio, fs)
    sf.write(audio_file_path, np.array(combined_audio).astype(np.int16), samplerate=SAMPLE_RATE, format="wav")

    merged_example["audio_filepath"] = audio_file_path
    # merged_example["original_audio_filepaths"] = "+".join([x["audio_filepath"] for x in examples])
    merged_example["original_audio_filepaths"] = output_filename

    return merged_example


def main(
    input_files: str,
    output_file: str,
    output_audio_dir: str,
    max_duration: int = 29,
    max_pause_beg_s: Union[int, float] = 0.04,
    min_pause_join_s: Union[int, float] = 0.05,
    max_pause_join_s: Union[int, float] = 0.2,
    max_pause_end_s: Union[int, float] = 0.04,
    audio_amplitude_normalization: int = 15_000,
    speaker_column_name: str = "speaker_id",
    num_workers: int = 32,
    max_samples: int = None,
):
    ext = input_files.rsplit(".", 1)[-1]
    input_files = input_files.split("+")
    input_files = input_files if len(input_files) > 1 else input_files[0]
    dataset = load_dataset(ext, data_files=input_files, split="train")
    # if max_samples is not None and max_samples < dataset.num_rows:
    #     dataset = dataset.select(range(max_samples))
    _print_ds_info(dataset)

    # more stricted on quality of transcriptions
    dataset = dataset.filter(lambda x: x["wer"] <= 10, num_proc=num_workers)
    _print_ds_info(dataset)

    # init or will take long time
    durations = dataset["duration"]
    languages = dataset["_language"]

    # shuffle indexes
    random.seed(10)
    indexes = random.sample(range(dataset.num_rows), dataset.num_rows)

    if max_samples is not None and max_samples < dataset.num_rows:
        indexes = indexes[:max_samples]

    # separate into groups no longer than max_duration
    cur_dur = 0
    merged_indexes = [[]]
    for idx in tqdm(indexes):
        if cur_dur + durations[idx] > max_duration:
            merged_indexes.append([idx])
            cur_dur = durations[idx]
        else:
            merged_indexes[-1].append(idx)
            cur_dur += durations[idx]
    # print(merged_indexes)
    print(f"Created {len(merged_indexes):,} groups")

    # remove monolingual groups (multi segments with same lang, or single segment)
    merged_indexes = [indexes_ for indexes_ in merged_indexes if len(set(languages[idx] for idx in indexes_)) > 1]
    print(f"Remaining {len(merged_indexes):,} groups after removing single-language groups")

    os.makedirs(output_audio_dir, exist_ok=True)

    def _process_func(indexes):
        return concat_examples(
            [dataset[idx] for idx in indexes],
            output_audio_dir,
            max_pause_beg_s,
            min_pause_join_s,
            max_pause_join_s,
            max_pause_end_s,
            audio_amplitude_normalization,
            speaker_column_name,
        )

    results = Parallel(n_jobs=num_workers)(
        delayed(_process_func)(
            indexes_,
        )
        for indexes_ in tqdm(merged_indexes)
    )

    # results = Parallel(n_jobs=num_workers)(
    #     delayed(concat_examples)(
    #         examples,
    #         output_audio_dir,
    #         max_pause_beg_s,
    #         min_pause_join_s,
    #         max_pause_join_s,
    #         max_pause_end_s,
    #         audio_amplitude_normalization,
            # speaker_column_name,
    #     )
    #     for examples in tqdm([[dataset[idx] for idx in indexes_] for indexes_ in merged_indexes])
    # )

    # todo
    # dataset = Dataset.from_list(results)

    # # add id column to keep segment order
    # dataset = dataset.map(
    #     lambda _, idx: {"id": f"{idx:09d}"},
    #     with_indices=True,
    #     num_proc=num_workers,
    #     desc="adding id column..."
    # )

    write_dataset_to_json(results, output_file_path=output_file, mode="w")


if __name__ == "__main__":
    fire.Fire(main)
