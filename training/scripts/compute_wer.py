#!/usr/bin/env python
# coding=utf-8

"""Normalize reference/hypothesis then compute WER."""

import sys
import os

# Get the parent directory of the scripts directory
parent_dir = os.path.dirname(os.path.abspath(os.path.dirname(__file__)))

# Add the parent directory to the system path
sys.path.append(parent_dir)

import json
import re

import evaluate
import fire
from datasets import load_dataset
from tqdm import tqdm

# from text_normalization.normalize_french import FrenchTextNormalizer
from normalizers import BasicTextNormalizer, EnglishTextNormalizer, FrenchTextNormalizer

# timestamp_pat = re.compile("\<[^\>]*\>")
timestamp_pat = re.compile(r"<\|(\d+\.\d+)\|>")


def write_dataset_to_json(dataset, output_file_path, mode="w", encoding="utf-8", default=str, ensure_ascii=False):
    ds_iter = iter(dataset)
    with open(output_file_path, mode, encoding=encoding) as fo:
        for _, sample in enumerate(tqdm(ds_iter, desc="Writing to json", total=len(dataset), unit=" samples")):
            fo.write(f"{json.dumps(sample, default=default, ensure_ascii=ensure_ascii)}\n")

    print(f"Saved manifest into {output_file_path}")


# Copied from transformers.models.whisper.tokenization_whisper.WhisperTokenizer._filter_timestamp_ids
def _filter_timestamps(token_ids):
    return re.sub(timestamp_pat, "", token_ids)


def main(
    input_file_path,
    output_file_path,
    text_column_name="text",
    language=None,
    num_workers=64,
):

    # tokenizer = WhisperTokenizerFast.from_pretrained(
    #     (model_args.tokenizer_name if model_args.tokenizer_name else model_args.model_name_or_path),
    #     cache_dir=model_args.cache_dir,
    #     use_fast=model_args.use_fast_tokenizer,
    #     revision=model_args.model_revision,
    #     token=token,
    # )

    # normalizer_ = FrenchTextNormalizer()
    normalizer_ = (
        EnglishTextNormalizer() if language == "en" else FrenchTextNormalizer() if language == "fr" else BasicTextNormalizer()
    )

    def normalizer(s):
        # remove timstamps
        s = _filter_timestamps(s)

        # normalize text
        # w/o "-"
        s = normalizer_(s, do_lowercase=True, do_ignore_words=False, symbols_to_keep="'", do_num2text=True)

        return s

    ext = input_file_path.rsplit(".", 1)[-1]
    dataset = load_dataset(ext, data_files=input_file_path, split="train")
    print(dataset)

    metric = evaluate.load("wer")

    def process_function(example):
        # normalize everything and re-compute the WER
        example["whisper_transcript_norm"] = normalizer(example["whisper_transcript"])
        example[f"{text_column_name}_norm"] = normalizer(example[text_column_name])

        example["whisper_transcript_wo_timestamp"] = _filter_timestamps(example["whisper_transcript"])

        example["wer_ortho"] = -1
        example["wer"] = -1
        if len(example[f"{text_column_name}_norm"]) > 0:
            # can't compute wer when reference=" "
            # example["wer_ortho"] = 100 * metric.compute(
            #     predictions=[example["whisper_transcript"]], references=[example[text_column_name]]
            # )
            example["wer_ortho"] = 100 * metric.compute(
                predictions=[example["whisper_transcript_wo_timestamp"]], references=[example[text_column_name]]
            )
            example["wer"] = 100 * metric.compute(
                predictions=[example["whisper_transcript_norm"]], references=[example[f"{text_column_name}_norm"]]
            )

        return example

    dataset = dataset.map(
        process_function,
        keep_in_memory=True,
        load_from_cache_file=False,
        num_proc=num_workers,
        desc="computing WER",
    )

    dataset = dataset.filter(
        lambda x: x["wer"] != -1,
        keep_in_memory=True,
        load_from_cache_file=False,
        num_proc=num_workers,
        desc="filtering wer==-1"
    )
    print(dataset)

    # wer_ortho = 100 * metric.compute(predictions=dataset["whisper_transcript"], references=dataset[text_column_name])
    wer_ortho = 100 * metric.compute(predictions=dataset["whisper_transcript_wo_timestamp"], references=dataset[text_column_name])
    wer = 100 * metric.compute(predictions=dataset["whisper_transcript_norm"], references=dataset[f"{text_column_name}_norm"])
    print(f"WER: {wer_ortho:.4f}%, Norm WER: {wer:.4f}%")

    # remove tmp col
    dataset = dataset.remove_columns("whisper_transcript_wo_timestamp")

    # export
    write_dataset_to_json(dataset, output_file_path=output_file_path, mode="w")


if __name__ == "__main__":
    fire.Fire(main)
