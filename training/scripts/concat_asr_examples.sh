#!/usr/bin/env bash
# Copyright 2023  Bofeng Huang

export HF_HOME="/projects/bhuang/.cache/huggingface"
export OMP_NUM_THREADS="1"
# export CUDA_VISIBLE_DEVICES="1,2,3,4,5"
export CUDA_VISIBLE_DEVICES=""

# input_file="/projects/bhuang/corpus/speech/nemo_manifests/mozilla-foundation/common_voice_17_0/fr/train/train_mozilla-foundation_common_voice_17_0_manifest_whisper_large_v3_wer.json"
input_file="/projects/bhuang/corpus/speech/nemo_manifests/mozilla-foundation/common_voice_17_0/fr/train/train_mozilla-foundation_common_voice_17_0_manifest_whisper_large_v3_wer_postminwer10.json"
# input_file="/projects/bhuang/corpus/speech/nemo_manifests/facebook/multilingual_librispeech/french/train/train_facebook_multilingual_librispeech_manifest.json"
# input_file="/projects/bhuang/corpus/speech/nemo_manifests/facebook/voxpopuli/fr/train/train_facebook_voxpopuli_manifest.json"
# input_file="/projects/bhuang/corpus/speech/nemo_manifests/espnet/yodas/fr000/train/train_espnet_yodas_manifest.json"

# output_file="${input_file%.*}_concatenated.json"
output_file="${input_file/\/train\//\/train_concatenated\/}"

#     --max_samples 1000 \

python scripts/concat_asr_examples.py \
    --input_file_path $input_file \
    --output_file_path $output_file \
    --preprocessing_batch_size 1000 \
    --preprocessing_num_workers 32
