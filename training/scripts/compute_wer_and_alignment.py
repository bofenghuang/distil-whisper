#!/usr/bin/env python
# Copyright 2023  Bofeng Huang

"""Normalize reference/hypothesis then compute WER."""

import sys
import os

# Get the parent directory of the scripts directory
parent_dir = os.path.dirname(os.path.abspath(os.path.dirname(__file__)))

# Add the parent directory to the system path
sys.path.append(parent_dir)

import json
import re
from typing import Optional

# from jiwer import wer
import fire
from datasets import load_dataset
from datasets.utils.logging import disable_progress_bar
from jiwer import process_words
from nltk import ngrams

# from .hf_dataset_processing.file_utils import write_dataset_to_json
from normalizers import BasicTextNormalizer, EnglishTextNormalizer, FrenchTextNormalizer
from asr_metric_calculation.compute_wer import compute_wer

disable_progress_bar()

# from nltk.stem.snowball import SnowballStemmer
# from nltk import word_tokenize
# Download the nltk data for tokenization
# import nltk
# nltk.download("punkt")
# Initialize SnowballStemmer with the specified language
# stemmer = SnowballStemmer("french")


# def stem_sentence(sentence):
#     # Tokenize the sentence into words
#     words = word_tokenize(sentence)

#     # Stem each word in the sentence
#     stemmed_words = [stemmer.stem(word) for word in words]

#     # Join the stemmed words back into a sentence
#     stemmed_sentence = " ".join(stemmed_words)

#     return stemmed_sentence


def compute_metrics(references, predictions, ngram_degree=None):
    wer_output = process_words(references, predictions)
    num_ref_words = sum([len(ref) for ref in wer_output.references])

    result = {
        "wer": 100 * wer_output.wer,
        "ier": 100 * wer_output.insertions / num_ref_words,
        "ser": 100 * wer_output.substitutions / num_ref_words,
        "der": 100 * wer_output.deletions / num_ref_words,
        "num_ref_words": num_ref_words,
        "num_ref_sentences": len(references),
    }

    if ngram_degree is not None:
        all_ngrams = list(ngrams(" ".join(predictions).split(), ngram_degree))
        repeated_ngrams_per_dataset = len(all_ngrams) - len(set(all_ngrams))

        repeated_ngrams_per_utt = 0
        for prediction_ in predictions:
            all_ngrams_ = list(ngrams(prediction_.split(), ngram_degree))
            repeated_ngrams_per_utt += len(all_ngrams_) - len(set(all_ngrams_))

        # result[f"repeated_{ngram_degree}grams"] = repeated_ngrams
        result[f"per_dataset_repeated_{ngram_degree}grams"] = repeated_ngrams_per_dataset
        result[f"per_utterance_repeated_{ngram_degree}grams"] = repeated_ngrams_per_utt

    return result


def main(
    input_file_path: str,
    output_dir: str,
    # suffix: str = "_wer",
    id_column_name: str = "_id",  # in fleurs, id can be repetitive
    target_column_name: str = "text",
    prediction_column_name: str = "prediction",
    language: Optional[str] = None,
    ngram_degree: Optional[int] = None,
    num_processing_workers: int = 1,
):
    ext = input_file_path.rsplit(".", 1)[-1]
    # output_file_path = f"{path}{suffix}.{ext}"
    dataset = load_dataset(ext, data_files=input_file_path, split="train")
    print(dataset)

    # Debug
    # dataset = dataset.select(range(1000))

    # normalizer = FrenchTextNormalizer()
    if language in ["en", "english"]:
        normalizer = EnglishTextNormalizer()
    # todo
    # elif language in ["fr", "french"]:
    #     normalizer = FrenchTextNormalizer()
    else:
        normalizer = BasicTextNormalizer()

    def normalize_(s):
        # s = re.sub(r"<[0-9\.]+>", "", s)  # remove timestamps
        s = normalizer(s)
        # s = normalizer(s, do_lowercase=True, do_ignore_words=False, symbols_to_keep="'", do_num2text=True)
        # s = normalizer(s, do_lowercase=True, do_ignore_words=False, symbols_to_keep="", do_num2text=True)
        return s

    def process_function(example, idx):
        example[f"{target_column_name}_norm"] = normalize_(example[target_column_name])
        example[f"{prediction_column_name}_norm"] = normalize_(example[prediction_column_name])

        example[f"{target_column_name}_split"] = example[target_column_name].split()
        example[f"{prediction_column_name}_split"] = example[prediction_column_name].split()
        example[f"{target_column_name}_norm_split"] = example[f"{target_column_name}_norm"].split()
        example[f"{prediction_column_name}_norm_split"] = example[f"{prediction_column_name}_norm"].split()

        # split into characters
        # norm_ = lambda s: " ".join(re.sub(r"\s+", "", s))
        # example[f"{target_column_name}_norm"] = norm_(example[f"{target_column_name}_norm"])
        # example[f"{prediction_column_name}_norm"] = norm_(example[f"{prediction_column_name}_norm"])

        # example["wer"] = wer(
        #     example[f"{target_column_name}_norm"], example[f"{prediction_column_name}_norm"]
        # )

        # example["stemmed_wer"] = wer(
        #     stem_sentence(example[f"{target_column_name}_norm"]),
        #     stem_sentence(example[f"{prediction_column_name}_norm"]),
        # )

        if id_column_name not in example:
            example[id_column_name] = f"{idx:09d}"

        return example

    dataset = dataset.map(
        process_function,
        with_indices=True,
        num_proc=num_processing_workers,
        # remove_columns=raw_datasets.column_names,
        load_from_cache_file=False,
        desc="normalizing...",
    )
    # print(dataset)

    # write_dataset_to_json(dataset, output_file_path)
    # print(f"The processed data is saved into {output_file_path}")

    # filter out empty targets
    dataset = dataset.filter(
        lambda x: len(x) > 0,
        input_columns=f"{target_column_name}_norm",
        num_proc=num_processing_workers,
        desc="filtering empty...",
    )

    # result = compute_metrics(dataset[target_column_name], dataset[prediction_column_name], ngram_degree=ngram_degree)
    # print("\nRAW METRICS")
    # print(json.dumps(result, indent=4))

    # result = compute_metrics(
    #     dataset[f"{target_column_name}_norm"], dataset[f"{prediction_column_name}_norm"], ngram_degree=ngram_degree
    # )
    # print("\nNORMALIZED METRICS")
    # print(json.dumps(result, indent=4))

    # speechbrain-style wer and alignment
    targets = dict(zip(dataset[id_column_name], dataset[f"{target_column_name}_split"]))
    predictions = dict(zip(dataset[id_column_name], dataset[f"{prediction_column_name}_split"]))
    compute_wer(targets, predictions, f"{output_dir}/wer_summary", do_print_top_wer=True, do_catastrophic=True)

    targets = dict(zip(dataset[id_column_name], dataset[f"{target_column_name}_norm_split"]))
    predictions = dict(zip(dataset[id_column_name], dataset[f"{prediction_column_name}_norm_split"]))
    compute_wer(targets, predictions, f"{output_dir}/wer_summary_normalized", do_print_top_wer=True, do_catastrophic=True)


if __name__ == "__main__":
    fire.Fire(main)
