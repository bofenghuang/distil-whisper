#!/usr/bin/env bash
# Copyright 2023  Bofeng Huang

myscriptspath="/home/bhuang/myscripts"
export PYTHONPATH="${PYTHONPATH:-}:$myscriptspath"
export PYTHONUNBUFFERED="1"

export HF_HOME="/projects/bhuang/.cache/huggingface"
export OMP_NUM_THREADS="1"
# export CUDA_VISIBLE_DEVICES="1,2,3,4,5"
# export CUDA_VISIBLE_DEVICES="0"

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

teacher_model_name_or_path="bofenghuang/whisper-large-v3-french"
model_name_or_path="./outputs/models/bofenghuang-whisper_large_v3_french_dec2_init"
output_dir="${model_name_or_path}_ft_ep8_lr1e4"

train_file="outputs/data/bofenghuang-whisper_large_v3_french/train-data-wer.json"
validation_file="/projects/bhuang/corpus/speech/nemo_manifests/final/2023-11-21/test_asr_mcv13_manifest_normalized_pnc.json+/projects/bhuang/corpus/speech/nemo_manifests/final/2023-11-21/test_asr_mls_manifest_normalized_pnc_cleaned.json+/projects/bhuang/corpus/speech/nemo_manifests/final/2023-11-21/test_asr_voxpopuli_manifest_normalized_pnc.json+/projects/bhuang/corpus/speech/nemo_manifests/final/2023-11-21/test_asr_fleurs_manifest_normalized_pnc.json+/projects/bhuang/corpus/speech/nemo_manifests/final/2023-11-21/test_asr_african_accented_french_manifest_normalized_pnc_cleaned.json+/projects/bhuang/corpus/speech/nemo_manifests/final/2023-11-21/test_asr_mtedx_manifest_normalized_pnc.json"

    # --weight_decay "0.01" \
    # --learning_rate 0.0001 \
    # --text_column_name "text" \
    # --text_column_name "whisper_transcript" \

# todo: teacher / student precision, text

accelerate launch \
    --multi_gpu \
    --num_processes=6 \
    run_distillation_b.py \
    --model_name_or_path "$model_name_or_path" \
    --teacher_model_name_or_path "$teacher_model_name_or_path" \
    --train_file "$train_file" \
    --validation_file "$validation_file" \
    --audio_column_name "audio_filepath" \
    --text_column_name "whisper_transcript" \
    --eval_text_column_name "text" \
    --max_duration_in_seconds 30 \
    --wer_threshold 10 \
    --timestamp_probability 0.2 \
    --condition_on_prev_probability 0 \
    --language "french" \
    --task "transcribe" \
    --preprocessing_num_workers 32 \
    --dataloader_num_workers 8 \
    --output_dir "$output_dir" \
    --overwrite_output_dir \
    --num_train_epochs "8" \
    --per_device_train_batch_size 32 \
    --per_device_eval_batch_size 16 \
    --gradient_accumulation_steps 1 \
    --kl_weight 1.0 \
    --temperature 2.0 \
    --optim "adamw_bnb_8bit" \
    --learning_rate 0.0001 \
    --warmup_ratio "0.05" \
    --lr_scheduler_type "cosine" \
    --dtype "float16" \
    --gradient_checkpointing \
    --freeze_encoder \
    --logging_steps 10 \
    --eval_steps 500 \
    --save_steps 500 \
    --save_total_limit 3 \
    --predict_with_generate \
    --generation_num_beams 1 \
    --max_label_length 128 \
    --return_timestamps False \
    --ddp_timeout 7200 \
    --wandb_project "hf-whisper-v3" \
    --do_train \
    --do_eval
