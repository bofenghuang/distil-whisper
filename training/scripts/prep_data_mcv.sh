#!/usr/bin/env bash
# Copyright 2023  Bofeng Huang

# prep data for mcv

set -x -e

echo "START TIME: $(date)"

# https://github.com/pytorch/audio/issues/1021#issuecomment-726915239
# export OMP_NUM_THREADS="1"

# cuda
export CUDA_VISIBLE_DEVICES="4,5,6,7"

# hf
export HF_HOME="/projects/bhuang/.cache/huggingface"
export TOKENIZERS_PARALLELISM="false"
# export BITSANDBYTES_NOWELCOME="1"
# export HF_HUB_ENABLE_HF_TRANSFER="1"
# export HF_HUB_OFFLINE="1"
# export HF_DATASETS_OFFLINE="1"
# export HF_EVALUATE_OFFLINE="1"

# Set your number of GPUs here
num_gpus=4
# CPUs
num_workers=256

#   --dtype "bfloat16" \
# --max_label_length 128 \
    # --max_samples_per_split 1024 \

CMD="accelerate launch --multi_gpu --num_processes=$num_gpus"

model_name_or_path="openai/whisper-large-v3"
# model_name_or_path="bofenghuang/whisper-large-v3-french"

input_file="/projects/bhuang/corpus/speech/nemo_manifests/mozilla-foundation/common_voice_17_0/fr/train/train_mozilla-foundation_common_voice_17_0_manifest.json"

# tmp_model_id="$(echo "${model_name_or_path}" | sed -e "s/-/\_/g" -e "s/[ |=/]/-/g")"
# outdir="./outputs/data/$tmp_model_id"

# tmp_model_id="$(echo "${model_name_or_path}" | sed -e "s/[ |=/-]/_/g")"
tmp_model_id="$(echo "${model_name_or_path##*/}" | sed -e "s/[ |=/-]/_/g")"
output_file="${input_file%.*}_${tmp_model_id}.json"

# pred
# $CMD run_pseudo_labelling_b.py \
#     --model_name_or_path "$model_name_or_path" \
#     --input_data_file "$input_file" \
#     --output_dir "$outdir" \
#     --output_data_file "$output_file" \
#     --audio_column_name "audio_filepath" \
#     --id_column_name "id" \
#     --duration_column_name "duration" \
#     --sort_by_duration True \
#     --preprocessing_num_workers $num_workers \
#     --pad_to_multiple_of 64 \
#     --dataloader_num_workers 8 \
#     --dtype "float16" \
#     --attn_implementation "flash_attention_2" \
#     --per_device_eval_batch_size 128 \
#     --language "fr" \
#     --task "transcribe" \
#     --return_timestamps \
#     --max_label_length 448 \
#     --generation_num_beams 1

# normalize (timestamps)
python scripts/norm_whisper_transcript.py \
    --input_file_path "$output_file" \
    --output_file_path "${output_file%.*}_norm.json" \
    --num_workers $num_workers

# wer
python scripts/compute_wer.py \
    --input_file_path "${output_file%.*}_norm.json" \
    --output_file_path "${output_file%.*}_norm_wer.json" \
    --num_workers $num_workers

# filter (upper-case, wer)
python scripts/filter_whisper_transcript.py \
    --input_file_path "${output_file%.*}_norm_wer.json" \
    --output_file_path "${output_file%.*}_norm_wer_filt.json" \
    --wer_threshold 10 \
    --num_workers $num_workers

input_file="${output_file%.*}_norm_wer_filt.json"
output_file="${input_file/\/train\//\/train_concatenated\/}"

# concat
python scripts/concat_asr_examples.py \
    --input_file_path $input_file \
    --output_file_path $output_file \
    --preprocessing_batch_size 1000 \
    --preprocessing_num_workers $num_workers

# wer
python scripts/compute_wer.py \
    --input_file_path "$output_file" \
    --output_file_path "${output_file%.*}_wer.json" \
    --num_workers $num_workers