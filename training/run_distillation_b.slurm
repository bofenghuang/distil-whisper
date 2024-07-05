#!/bin/bash
# Copyright 2023  Bofeng Huang

#SBATCH --job-name=training
#SBATCH --output=logs/%x/%j.out # output file (%j = job ID)
#SBATCH --error=logs/%x/%j.err # error file (%j = job ID)
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1          # crucial - only 1 task per dist per node!
#SBATCH --cpus-per-task=20         # number of cores per tasks
#SBATCH --gres=gpu:8 # reserve 8 GPUs per node
#SBATCH --time 100:00:00              # maximum execution time (HH:MM:SS)
#SBATCH --qos=qos_gpu-t4             # QoS
#SBATCH --hint=nomultithread         # we get physical cores not logical
#SBATCH --partition=gpu_p2
#SBATCH --account=cjc@v100

# Run distillation

set -x -e

echo "START TIME: $(date)"

# set up environment
module purge
module load git-lfs
module load unrar
module load anaconda-py3/2023.03
module load cuda/12.1.0
# conda activate speech
conda activate asr

# Debugging flags (optional)
# force crashing on nccl issues like hanging broadcast
# export NCCL_ASYNC_ERROR_HANDLING=1
# export NCCL_DEBUG=INFO
# export NCCL_DEBUG_SUBSYS=COLL
# export NCCL_SOCKET_NTHREADS=1
# export NCCL_NSOCKS_PERTHREAD=1
# export CUDA_LAUNCH_BLOCKING=1
# export PYTHONFAULTHANDLER=1

# https://github.com/pytorch/audio/issues/1021#issuecomment-726915239
# export OMP_NUM_THREADS="1"

# cuda
# export CUDA_VISIBLE_DEVICES="4,5,6,7"

# hf
# export HF_HOME="/projects/bhuang/.cache/huggingface"
export TOKENIZERS_PARALLELISM="false"
# export BITSANDBYTES_NOWELCOME="1"
# export HF_HUB_ENABLE_HF_TRANSFER="1"
export HF_HUB_OFFLINE="1"
export HF_DATASETS_OFFLINE="1"
export HF_EVALUATE_OFFLINE="1"

# wandb
export WANDB_MODE=offline
# export WANDB_DISABLED=true
# export WANDB_API_KEY=YOUR_WANDB_API_KEY
# export WANDB_ENTITY=YOUR_WANDB_ENTITY
export WANDB_PROJECT=hf-whisper-v4.1

# Set your number of GPUs here
# num_gpus=4
num_gpus=8
# CPUs
num_workers=64

# cmd
# CMD="accelerate launch --num_processes=1"
CMD="accelerate launch --multi_gpu --num_processes=$num_gpus"

# teacher_model_name_or_path="openai/whisper-large-v3"
# teacher_model_name_or_path="bofenghuang/whisper-large-v3-french"
teacher_model_name_or_path="/gpfswork/rech/cjc/commun/models/whisper/whisper-large-v3"

# model_name_or_path="/projects/bhuang/models/asr/whisper/openai-whisper_large_v3_dec2_init"
model_name_or_path="/gpfswork/rech/cjc/commun/models/whisper/whisper-large-v3_dec2_init"
# model_name_or_path="distil-whisper/distil-large-v3"

tmp_model_id="$(echo "${model_name_or_path##*/}" | sed -e "s/[ |=/-]/_/g")"
# output_dir="./outputs/french/${tmp_model_id}_ft_ep20_bs256_lr1e4"
output_dir="${model_name_or_path}_ft_ep20_bs256_lr1e4"
wandb_run_name="${output_dir##*/}"

train_file="/gpfsscratch/rech/cjc/commun/corpus/speech/nemo_manifests/mozilla-foundation/common_voice_17_0/it/train_concatenated/train_mozilla-foundation_common_voice_17_0_manifest_whisper_large_v3_norm_wer_filt_wer.json"
validation_file="/gpfsscratch/rech/cjc/commun/corpus/speech/nemo_manifests/mozilla-foundation/common_voice_17_0/it/validation/validation_mozilla-foundation_common_voice_17_0_manifest.json+/gpfsscratch/rech/cjc/commun/corpus/speech/nemo_manifests/google/fleurs/it_it/validation/validation_google_fleurs_manifest.json"

    # --max_train_samples 512 \
    # --max_eval_samples 512 \
    # --max_eval_samples 2000 \
    # --language "french" \
    # --ddp_timeout 7200 \
    # --weight_decay "0.01" \
    # --attn_implementation "flash_attention_2" \
    # --attn_implementation "sdpa" \
    # --gradient_checkpointing \
    # --overwrite_output_dir \

$CMD run_distillation_b.py \
    --model_name_or_path "$model_name_or_path" \
    --teacher_model_name_or_path "$teacher_model_name_or_path" \
    --apply_spec_augment \
    --train_file "$train_file" \
    --validation_file "$validation_file" \
    --max_eval_samples 2000 \
    --audio_column_name "audio_filepath" \
    --text_column_name "whisper_transcript" \
    --eval_text_column_name "text" \
    --duration_column_name "duration" \
    --max_duration_in_seconds 30 \
    --min_duration_in_seconds 0 \
    --prev_text_column_name "prev_whisper_transcript" \
    --condition_on_prev_column_name "condition_on_prev" \
    --language_column_name "_language" \
    --max_label_length 448 \
    --wer_threshold 10 \
    --timestamp_probability 0.5 \
    --condition_on_prev_probability 0.5 \
    --task "transcribe" \
    --preprocessing_num_workers $num_workers \
    --dataloader_num_workers 8 \
    --pad_to_multiple_of 64 \
    --output_dir "$output_dir" \
    --overwrite_output_dir \
    --num_train_epochs 20 \
    --per_device_train_batch_size 64 \
    --per_device_eval_batch_size 32 \
    --gradient_accumulation_steps 1 \
    --kl_weight 1.0 \
    --temperature 2.0 \
    --learning_rate "1e-4" \
    --adam_beta1 0.9 \
    --adam_beta2 0.95 \
    --adam_epsilon "1e-5" \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --dtype "bfloat16" \
    --gradient_checkpointing \
    --attn_implementation "sdpa" \
    --freeze_encoder \
    --freeze_embed_positions \
    --logging_steps 10 \
    --eval_steps 100 \
    --save_steps 100 \
    --save_total_limit 3 \
    --predict_with_generate \
    --generation_num_beams 1 \
    --return_timestamps False \
    --wandb_project $WANDB_PROJECT \
    --wandb_run_name $wandb_run_name \
    --do_train \
    --do_eval

echo "END TIME: $(date)"
