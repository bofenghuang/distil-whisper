#!/usr/bin/env bash
# Copyright 2023  Bofeng Huang

set -x -e

# Python path
myscriptspath="/home/bhuang/myscripts"
export PYTHONPATH="${PYTHONPATH:-}:$myscriptspath"
export PYTHONUNBUFFERED="1"

# HF cache
export HF_HOME="/projects/bhuang/.cache/huggingface"

# WANDB related
# export WANDB_MODE=offline
# export WANDB_DISABLED=true
# export WANDB_API_KEY=YOUR_WANDB_API_KEY
# export WANDB_ENTITY=YOUR_WANDB_ENTITY
export WANDB_PROJECT=hf-whisper-v3

# https://github.com/pytorch/audio/issues/1021#issuecomment-726915239
export OMP_NUM_THREADS="1"

# export CUDA_VISIBLE_DEVICES="1,2,3,4,5"
# export CUDA_VISIBLE_DEVICES="0"

# Debugging flags (optional)
# force crashing on nccl issues like hanging broadcast
# export NCCL_ASYNC_ERROR_HANDLING=1
# export NCCL_DEBUG=INFO
# export NCCL_DEBUG_SUBSYS=COLL
# export NCCL_SOCKET_NTHREADS=1
# export NCCL_NSOCKS_PERTHREAD=1
# export CUDA_LAUNCH_BLOCKING=1
# export PYTHONFAULTHANDLER=1

# We found that upwards of 13k hours of audio data was required to reach convergence on English ASR (see Section 9.2 of the paper), so the more data you have, the better!

# lr_scheduler_type
# When experimenting with a training set-up or training for very few steps (< 5k),
# using constant_with_warmup is typically beneficial, since the learning rate remains high over the short training run.
# When performing long training runs (> 5k), using a linear schedule generally results in superior downstream performance of the distilled model.

# timestamp_probability
# The per-sample probability for retaining timestamp tokens in the labels (should they contain them).
# Retaining some portion of timestamp tokens in the training data is required to ensure the distilled model can predict timestamps at inference time.
# In our experiments, we found that training on timestamps with high-probability hurts the distilled model's transcription performance.
# Thus, we recommend setting this to a value below 0.5. Typically, a value of 0.2 works well, giving good transcription and timestamp performance.

# condition_on_prev_probability
# The per-sample probability for conditioning on previous labels.
# Conditioning on previous tokens is required to ensure the distilled model can be used with the "sequential" long-form transcription algorithm at inference time.
# We did not experiment with this parameter, but found a value of 0.1 to provide adequate performance. OpenAI pre-trained Whisper on with a 50% probability for conditioning on previous tokens. Thus, you might wish to try higher values.

# Pseudo-labels can be used when either:
# The original text transcriptions are normalised (lower-cased or no punctuation): the Whisper generated pseudo-labels contain both punctuation and casing, and so can be used as a substitute for the normalised transcriptions
# The pre-trained Whisper model achieves < 20% WER on the languages: we then know the majority of the pseudo-labels will be accurate enough for us to train on.

# use_cache
# set to false to save vram

teacher_model_name_or_path="bofenghuang/whisper-large-v3-french"
model_name_or_path="./outputs/models/bofenghuang-whisper_large_v3_french_dec2_init"
output_dir="${model_name_or_path}_ft_ep16_bs256_lr1e4_preprend"
wandb_run_name="${output_dir##*/}"

train_file="/projects/bhuang/corpus/speech/nemo_manifests/final/2023-11-21/train-data-wer.json"
validation_file="/projects/bhuang/corpus/speech/nemo_manifests/final/2023-11-21/test_asr_mcv13_manifest_normalized_pnc.json+/projects/bhuang/corpus/speech/nemo_manifests/final/2023-11-21/test_asr_mls_manifest_normalized_pnc_cleaned.json+/projects/bhuang/corpus/speech/nemo_manifests/final/2023-11-21/test_asr_voxpopuli_manifest_normalized_pnc.json+/projects/bhuang/corpus/speech/nemo_manifests/final/2023-11-21/test_asr_fleurs_manifest_normalized_pnc.json+/projects/bhuang/corpus/speech/nemo_manifests/final/2023-11-21/test_asr_african_accented_french_manifest_normalized_pnc_cleaned.json"

    # --weight_decay "0.01" \
    # --learning_rate 0.0001 \

    # --max_train_samples 512 \
    # --max_eval_samples 512 \

    # --text_column_name "text" \
    # --text_column_name "whisper_transcript" \

# todo: teacher / student precision, text
    # --overwrite_output_dir \

accelerate launch \
    --multi_gpu \
    --num_processes=4 \
    run_distillation_b.py \
    --model_name_or_path "$model_name_or_path" \
    --teacher_model_name_or_path "$teacher_model_name_or_path" \
    --apply_spec_augment \
    --train_file "$train_file" \
    --validation_file "$validation_file" \
    --audio_column_name "audio_filepath" \
    --text_column_name "whisper_transcript" \
    --eval_text_column_name "text" \
    --max_duration_in_seconds 30 \
    --wer_threshold 10 \
    --timestamp_probability 0.2 \
    --condition_on_prev_probability 0.1 \
    --language "french" \
    --task "transcribe" \
    --preprocessing_num_workers 16 \
    --dataloader_num_workers 8 \
    --output_dir "$output_dir" \
    --overwrite_output_dir \
    --num_train_epochs 16 \
    --per_device_train_batch_size 32 \
    --per_device_eval_batch_size 16 \
    --gradient_accumulation_steps 2 \
    --kl_weight 1.0 \
    --temperature 2.0 \
    --optim "adamw_bnb_8bit" \
    --learning_rate 0.0001 \
    --warmup_steps 500 \
    --lr_scheduler_type "cosine" \
    --dtype "float16" \
    --gradient_checkpointing \
    --freeze_encoder \
    --logging_steps 10 \
    --eval_steps 1000 \
    --save_steps 1000 \
    --save_total_limit 3 \
    --predict_with_generate \
    --generation_num_beams 1 \
    --max_label_length 256 \
    --return_timestamps False \
    --ddp_timeout 7200 \
    --wandb_project $WANDB_PROJECT \
    --wandb_run_name $wandb_run_name \
    --do_train \
    --do_eval
