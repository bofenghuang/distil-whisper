#!/usr/bin/env python
# coding=utf-8

import json
import re

import evaluate
import fire
from datasets import load_dataset
from tqdm import tqdm

from text_normalization.normalize_french import FrenchTextNormalizer

timestamp_pat = re.compile(r"<\|(\d+\.\d+)\|>")


def write_dataset_to_json(dataset, output_file_path, mode="w", default=str, ensure_ascii=False):
    ds_iter = iter(dataset)
    with open(output_file_path, mode) as fo:
        for _, sample in enumerate(tqdm(ds_iter, desc="Writing to json", total=len(dataset), unit=" samples")):
            fo.write(f"{json.dumps(sample, default=default, ensure_ascii=ensure_ascii)}\n")


# Copied from transformers.models.whisper.tokenization_whisper.WhisperTokenizer._filter_timestamp_ids
def _filter_timestamp_ids(token_ids):
    return re.sub(timestamp_pat, "", token_ids)


def main(
    dataset_file,
    final_output_json,
    text_column_name="text",
    num_workers=64,
):

    # tokenizer = WhisperTokenizerFast.from_pretrained(
    #     (model_args.tokenizer_name if model_args.tokenizer_name else model_args.model_name_or_path),
    #     cache_dir=model_args.cache_dir,
    #     use_fast=model_args.use_fast_tokenizer,
    #     revision=model_args.model_revision,
    #     token=token,
    # )

    normalizer_ = FrenchTextNormalizer()

    def normalizer(s):
        # s = re.sub(r"<\|[0-9\.]+\|>", "", s)  # remove timstamps
        # s = re.sub(r"\<[^\>]*\>", "", s)  # remove timstamps
        s = _filter_timestamp_ids(s)
        s = normalizer_(s, do_lowercase=True, do_ignore_words=False, symbols_to_keep="'", do_num2text=True)  # w/o "-"

        return s

    ext = dataset_file.rsplit(".", 1)[-1]
    dataset = load_dataset(ext, data_files=dataset_file, split="train")
    print(dataset)

    metric = evaluate.load("wer")

    def process_function(example):
        # todo: we assume here both pred and label are string
        # example["whisper_transcript"] = id_pred_mappings[example[id_column_name]]
        example["wer_ortho"] = 100 * metric.compute(
            predictions=[example["whisper_transcript"]], references=[example[text_column_name]]
        )

        # normalize everything and re-compute the WER
        example["whisper_transcript_norm"] = normalizer(example["whisper_transcript"])
        example[f"{text_column_name}_norm"] = normalizer(example[text_column_name])

        example["wer"] = -1
        if len(example[f"{text_column_name}_norm"]) > 0:
            example["wer"] = 100 * metric.compute(
                predictions=[example["whisper_transcript_norm"]], references=[example[f"{text_column_name}_norm"]]
            )

        return example

    dataset = dataset.map(process_function, num_proc=num_workers, desc="computing WER")
    dataset = dataset.filter(lambda x: x["wer"] != -1, num_proc=num_workers)
    print(dataset)

    write_dataset_to_json(dataset, output_file_path=final_output_json, mode="w")

    wer_ortho = 100 * metric.compute(predictions=dataset["whisper_transcript"], references=dataset[text_column_name])
    wer = 100 * metric.compute(predictions=dataset["whisper_transcript_norm"], references=dataset[f"{text_column_name}_norm"])
    print(f"WER: {wer_ortho:.4f}%, Norm WER: {wer:.4f}%")


if __name__ == "__main__":
    fire.Fire(main)
