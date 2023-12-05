#!/usr/bin/env bash
# Copyright 2023  Bofeng Huang

myscriptspath="/home/bhuang/myscripts"
export PYTHONPATH="${PYTHONPATH:-}:$myscriptspath"
export PYTHONUNBUFFERED="1"

export HF_HOME="/projects/bhuang/.cache/huggingface"
export OMP_NUM_THREADS="1"
# export CUDA_VISIBLE_DEVICES="1,2,3,4,5"
# export CUDA_VISIBLE_DEVICES="0"

# return_timestamps
# Timestamp prediction is required should you want your distilled model to be able to predict timestamps
# at inference time (e.g. for the original OpenAI long-form transcription algorithm).
# However, the pseudo-labels are marginally less accurate than not using timestamps.
# We recommend pseudo-labelling with timestamps to ensure the distilled model is as general as possible.

# decode_token_ids
# The Whisper tokenizer does not preserve the same token ids across encoding/decoding steps, that is encode(decode(token_ids)) != token_ids.

#   --report_to "wandb" \
#   --wandb_project "distil-whisper-labelling" \
#   --push_to_hub
#   --dtype "bfloat16" \
    # --max_label_length 128 \

CMD="accelerate launch --multi_gpu --num_processes=6"

model_name_or_path="bofenghuang/whisper-large-v3-french"

train_file="/projects/bhuang/corpus/speech/nemo_manifests/final/2023-11-21/train_asr_processed_cleaned.json"

tmp_model_id="$(echo "${model_name_or_path}" | sed -e "s/-/\_/g" -e "s/[ |=/]/-/g")"
outdir="./outputs/data/$tmp_model_id"

    # --max_samples_per_split 100000 \

$CMD run_pseudo_labelling_b.py \
    --model_name_or_path "$model_name_or_path" \
    --dataset_file "$train_file" \
    --output_dir "$outdir" \
    --audio_column_name "audio_filepath" \
    --text_column_name "text" \
    --id_column_name "id" \
    --duration_column_name "duration" \
    --preprocessing_num_workers 64 \
    --streaming False \
    --dataloader_num_workers 8 \
    --dtype "float16" \
    --attn_type "flash_attn" \
    --language "fr" \
    --task "transcribe" \
    --return_timestamps \
    --per_device_eval_batch_size 32 \
    --max_label_length 448 \
    --generation_num_beams 1 \
    --decode_token_ids True \
    --logging_steps 500

python compute_wer.py \
    --dataset_file "${outdir}/train-data.json" \
    --final_output_json "${outdir}/train-data-wer.json" \
    --text_column_name "text" \
    --num_workers 64
