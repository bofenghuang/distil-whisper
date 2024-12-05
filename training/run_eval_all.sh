#!/usr/bin/env bash
# Copyright 2024  Bofeng Huang

# export CUDA_VISIBLE_DEVICES="7"

model_names_or_paths=(
    $1
    # "openai/whisper-large-v2"
    # "openai/whisper-large-v3"
    # "openai/whisper-large-v3-turbo"
    # "distil-whisper/distil-large-v3"
    # "eustlb/distil-large-v3-fr"
    # "bofenghuang/whisper-large-v3-french"
    # "bofenghuang/whisper-large-v3-french-distil-dec16"
    # "bofenghuang/whisper-large-v3-french-distil-dec2"
    # "bofenghuang/whisper-large-v3-distil-fr-v0.2"
    # "bofenghuang/whisper-large-v3-distil-it-v0.2"
    # "/projects/bhuang/models/asr/whisper/fr/whisper-large-v3-distil-fr"
    # "/projects/bhuang/models/asr/whisper/multi/whisper-large-v3-distil-multi-exp2/whisper_large_v3_turbo_repl_enc_dec2_init_cs_ft_multi20x46k_en10x46k_cs10x9k_ep20_bs2048_lr3e4_specaugxtime022x30x2xfeat022x14x2_bpedropout005_condonprev02_linear"
    # "/projects/bhuang/models/asr/whisper/multi/whisper-large-v3-distil-multi-exp2/whisper_large_v3_dec2_init_cs_ft_multi20x46k_en10x98k_cs10x9k_ep20_bs2048_lr3e4_specaugxtime03x10x2xfeat022x14x2_bpedropout005_condonprev02_linear"
)

for model_name_or_path in "${model_names_or_paths[@]}"; do
    for lang in en it fr es pt de nl; do
    # for lang in fr; do
    # for lang in it; do
    # for lang in en; do
    # for lang in $2; do
    # for lang in de; do
        ./run_eval_b2.sh $model_name_or_path $lang
    done
done
