#!/usr/bin/env bash
# Copyright 2023  Bofeng Huang

export HF_HOME="/projects/bhuang/.cache/huggingface"
export OMP_NUM_THREADS="1"
export CUDA_VISIBLE_DEVICES="4,5"

# return_timestamps
# Timestamp prediction is required should you want your distilled model to be able to predict timestamps
# at inference time (e.g. for the original OpenAI long-form transcription algorithm).
# However, the pseudo-labels are marginally less accurate than not using timestamps.
# We recommend pseudo-labelling with timestamps to ensure the distilled model is as general as possible.

# decode_token_ids
# The Whisper tokenizer does not preserve the same token ids across encoding/decoding steps, that is encode(decode(token_ids)) != token_ids.

#   --dtype "bfloat16" \
# --max_label_length 128 \
    # --max_samples_per_split 1024 \

CMD="accelerate launch --multi_gpu --num_processes=2"

model_name_or_path="openai/whisper-large-v3"
# model_name_or_path="bofenghuang/whisper-large-v3-french"

input_data_file="/projects/bhuang/corpus/speech/nemo_manifests/mozilla-foundation/common_voice_17_0/fr/train/train_mozilla-foundation_common_voice_17_0_manifest.json"

# tmp_model_id="$(echo "${model_name_or_path}" | sed -e "s/-/\_/g" -e "s/[ |=/]/-/g")"
# outdir="./outputs/data/$tmp_model_id"

# tmp_model_id="$(echo "${model_name_or_path}" | sed -e "s/[ |=/-]/_/g")"
tmp_model_id="$(echo "${model_name_or_path##*/}" | sed -e "s/[ |=/-]/_/g")"
output_file="${input_data_file%.*}_${tmp_model_id}.json"
final_output_file="${output_file%.*}_wer.json"

$CMD run_pseudo_labelling_b.py \
    --model_name_or_path "$model_name_or_path" \
    --input_data_file "$input_data_file" \
    --max_samples_per_split 1024 \
    --output_dir "$outdir" \
    --output_data_file "$output_file" \
    --audio_column_name "audio_filepath" \
    --id_column_name "id" \
    --duration_column_name "duration" \
    --sort_by_duration True \
    --preprocessing_num_workers 64 \
    --pad_to_multiple_of 64 \
    --dataloader_num_workers 8 \
    --dtype "float16" \
    --attn_implementation "flash_attention_2" \
    --per_device_eval_batch_size 128 \
    --language "fr" \
    --task "transcribe" \
    --return_timestamps \
    --max_label_length 448 \
    --generation_num_beams 1

python compute_wer.py \
    --input_data_file "$output_file" \
    --output_data_file "$final_output_file" \
    --num_workers 64
