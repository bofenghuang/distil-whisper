#!/usr/bin/env bash
# Copyright 2023  Bofeng Huang

# Concat utterances

set -x -e

echo "START TIME: $(date)"

# https://github.com/pytorch/audio/issues/1021#issuecomment-726915239
# export OMP_NUM_THREADS="1"

# cuda
# export CUDA_VISIBLE_DEVICES=""

# hf
export HF_HOME="/projects/bhuang/.cache/huggingface"
export TOKENIZERS_PARALLELISM="false"
# export BITSANDBYTES_NOWELCOME="1"
# export HF_HUB_ENABLE_HF_TRANSFER="1"
# export HF_HUB_OFFLINE="1"
# export HF_DATASETS_OFFLINE="1"
# export HF_EVALUATE_OFFLINE="1"

# CPUs
num_workers=128

# input_file="/projects/bhuang/corpus/speech/nemo_manifests/mozilla-foundation/common_voice_17_0/fr/train/train_mozilla-foundation_common_voice_17_0_manifest_whisper_large_v3_wer.json"
input_file="/projects/bhuang/corpus/speech/nemo_manifests/mozilla-foundation/common_voice_17_0/fr/train/train_mozilla-foundation_common_voice_17_0_manifest_whisper_large_v3_wer_postminwer10.json"
# input_file="/projects/bhuang/corpus/speech/nemo_manifests/facebook/multilingual_librispeech/french/train/train_facebook_multilingual_librispeech_manifest.json"
# input_file="/projects/bhuang/corpus/speech/nemo_manifests/facebook/voxpopuli/fr/train/train_facebook_voxpopuli_manifest.json"
# input_file="/projects/bhuang/corpus/speech/nemo_manifests/espnet/yodas/fr000/train/train_espnet_yodas_manifest.json"

# take arg
input_file=$1

input_dir="${input_file%/*}"
# output_file="${input_file%.*}_concatenated.json"
output_file="${input_file/\/train\//\/train_concatenated\/}"

#     --max_samples 1000 \

python scripts/concat_asr_examples.py \
    --input_file_path $input_file \
    --output_file_path $output_file \
    --preprocessing_batch_size 1000 \

# tmp: remove unconcatenated audio files
find $input_dir -mindepth 1 -type f -name "*.wav" -delete
find $input_dir -mindepth 1 -type d -empty -delete

echo "END TIME: $(date)"
