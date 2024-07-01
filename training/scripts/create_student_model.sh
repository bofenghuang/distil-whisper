#!/usr/bin/env bash
# Copyright 2023  Bofeng Huang

export HF_HOME="/projects/bhuang/.cache/huggingface"
export OMP_NUM_THREADS="1"


model_name_or_path="bofenghuang/whisper-large-v3-french"

tmp_model_id="$(echo "${model_name_or_path}" | sed -e "s/-/\_/g" -e "s/[ |=/]/-/g")"
save_dir="./outputs/models/${tmp_model_id}_dec2_init"

python create_student_model.py \
    --teacher_checkpoint "$model_name_or_path" \
    --decoder_layers 2 \
    --save_dir "$save_dir"
