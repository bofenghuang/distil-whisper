#!/bin/bash
# Copyright 2023  Bofeng Huang

#SBATCH --job-name=d-mono
#SBATCH --output=logs/%x/%j.out      # output file (%j = job ID)
#SBATCH --error=logs/%x/%j.err       # error file (%j = job ID)
#SBATCH --nodes=8
#SBATCH --exclusive
#SBATCH --ntasks-per-node=1          # crucial - only 1 task per dist per node!
#SBATCH --cpus-per-task=96           # number of cores per tasks
#SBATCH --gres=gpu:4                 # reserve number of GPUs per node
#SBATCH --time=20:00:00              # maximum execution time (HH:MM:SS)
#SBATCH --hint=nomultithread         # we get physical cores not logical
#SBATCH --constraint=h100
#SBATCH --account=gkb@h100           # account to use, with xyz the 3 letter code of your project

# Run distillation on monolingual

set -x -e

echo "START TIME: $(date)"

# set up environment
module purge
module load arch/h100
module load git-lfs
# module load unrar
module load anaconda-py3/2024.06
module load cuda/12.2.0
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

# export TORCH_CPP_LOG_LEVEL=INFO
# export TORCH_DISTRIBUTED_DEBUG=INFO
# export TORCH_SHOW_CPP_STACKTRACES=1

# https://github.com/pytorch/audio/issues/1021#issuecomment-726915239
# export OMP_NUM_THREADS="1"

# slurm buffers python output by default, unbuffer python -u
export PYTHONUNBUFFERED="1"

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
# export WANDB_PROJECT=hf-whisper-v4.1
# export WANDB_PROJECT=hf-whisper-v4.2
# export WANDB_PROJECT=hf-whisper-v4.3
export WANDB_PROJECT=hf-whisper-v4.4

# Set CPUs
num_workers=64

# Set GPUs
export GPUS_PER_NODE=4

# Set network
head_node_ip=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)

# export LAUNCHER="accelerate launch --num_processes=1"
# export LAUNCHER="accelerate launch --multi_gpu --num_processes=$GPUS_PER_NODE"
    # --deepspeed_config_file deepspeed_configs/ds_config_zero2_no_offload.json
export LAUNCHER="accelerate launch \
    --num_processes $((SLURM_NNODES * GPUS_PER_NODE)) \
    --num_machines $SLURM_NNODES \
    --rdzv_backend c10d \
    --main_process_ip $head_node_ip \
    --main_process_port 29500 \
    --machine_rank $SLURM_NODEID \
    "

# teacher_model_name_or_path="openai/whisper-large-v3"
# teacher_model_name_or_path="bofenghuang/whisper-large-v3-french"
# teacher_model_name_or_path="/lustre/fswork/projects/rech/cjc/commun/models/whisper/whisper-large-v3"
teacher_model_name_or_path="/lustre/fswork/projects/rech/gkb/commun/models/pretrained/whisper/whisper-large-v3"

# model_name_or_path="/projects/bhuang/models/asr/whisper/openai-whisper_large_v3_dec2_init"
# model_name_or_path="/lustre/fswork/projects/rech/gkb/commun/models/pretrained/whisper/distil-large-v3"
model_name_or_path="/lustre/fswork/projects/rech/gkb/commun/models/pretrained/whisper/whisper-large-v3_dec2_init"
# model_name_or_path="/lustre/fswork/projects/rech/gkb/commun/models/pretrained/whisper/whisper-large-v3-turbo_dec2_init"
# model_name_or_path="/lustre/fswork/projects/rech/gkb/commun/models/pretrained/whisper/whisper-large-v3-turbo-repl-enc"
# model_name_or_path="/lustre/fswork/projects/rech/gkb/commun/models/pretrained/whisper/whisper-large-v3-turbo-repl-enc_dec2_init"

tmp_model_id="$(echo "${model_name_or_path##*/}" | sed -e "s/[ |=/-]/_/g")"
# output_root="/lustre/fswork/projects/rech/cjc/commun/outputs/whisper/fr"
output_root="/lustre/fswork/projects/rech/gkb/commun/outputs/whisper/de"
output_dir="${output_root}/${tmp_model_id}_ft_filt20_ep160_bs2048_lr3e4_specaugxtime022x30x2xfeat022x14x2_bpedropout005_condonprev02_linear"
wandb_run_name="${output_dir##*/}"

# input_root=/projects/bhuang/corpus/speech/nemo_manifests
input_root=/lustre/fsn1/projects/rech/gkb/commun/corpus/speech/stt-pseudo-labeled-whisper-large-v3-multilingual

# it
# train_files=(
#     "${input_root}/mozilla-foundation/common_voice_17_0/it/train_concatenated/train_mozilla-foundation_common_voice_17_0_manifest_whisper_large_v3_norm_wer_filt_wer_zipped.json"
#     "${input_root}/facebook/multilingual_librispeech/italian/train_concatenated/train_facebook_multilingual_librispeech_manifest_whisper_large_v3_norm_upprev_wer_filt_zipped.json"
#     "${input_root}/facebook/voxpopuli/it/train_concatenated/train_facebook_voxpopuli_manifest_whisper_large_v3_norm_upprev_wer_filt_zipped.json"
#     "${input_root}/multilingual-tedx/it-it/train_concatenated/train_asr_whisper_large_v3_norm_upprev_wer_filt_zipped.json"
#     "${input_root}/espnet/yodas/it000/train_concatenated/train_espnet_yodas_manifest_whisper_large_v3_merged_norm_upprev_wer_filt_zipped.json"
#     "${input_root}/espnet/yodas/it100/train_concatenated/train_espnet_yodas_manifest_whisper_large_v3_merged_norm_upprev_wer_filt_zipped.json"
#     "${input_root}/espnet/yodas/it101/train_concatenated/train_espnet_yodas_manifest_whisper_large_v3_merged_norm_upprev_wer_filt_zipped.json"
# )
# validation_files=(
#     "${input_root}/mozilla-foundation/common_voice_17_0/it/validation/validation_mozilla-foundation_common_voice_17_0_manifest.json"
#     "${input_root}/facebook/multilingual_librispeech/italian/validation/validation_facebook_multilingual_librispeech_manifest.json"
#     # "${input_root}/facebook/voxpopuli/it/validation/validation_facebook_voxpopuli_manifest.json"
#     "${input_root}/google/fleurs/it_it/validation/validation_google_fleurs_manifest.json"
# )

# fr
# train_files=(
#     "${input_root}/mozilla-foundation/common_voice_17_0/fr/train_concatenated/train_mozilla-foundation_common_voice_17_0_manifest_whisper_large_v3_norm_wer_filt_zipped_wer.json"
#     "${input_root}/facebook/multilingual_librispeech/french/train_concatenated/train_facebook_multilingual_librispeech_manifest_zipped_whisper_large_v3_norm_upprev_wer_filt.json"
#     "${input_root}/facebook/voxpopuli/fr/train_concatenated/train_facebook_voxpopuli_manifest_zipped_whisper_large_v3_norm_upprev_wer_filt.json"
#     "${input_root}/multilingual-tedx/fr-fr/train_concatenated/train_asr_zipped_whisper_large_v3_norm_upprev_wer_filt.json"
#     "${input_root}/gigant/african_accented_french/train_concatenated/train_gigant_african_accented_french_manifest_whisper_large_v3_norm_wer_filt_zipped_wer.json"
#     "${input_root}/espnet/yodas/fr000/train_concatenated/train_espnet_yodas_manifest_zipped_whisper_large_v3_norm_upprev_wer_filt.json"
#     "${input_root}/espnet/yodas/fr100/train_concatenated/train_espnet_yodas_manifest_zipped_whisper_large_v3_norm_upprev_wer_filt.json"
#     "${input_root}/espnet/yodas/fr101/train_concatenated/train_espnet_yodas_manifest_zipped_whisper_large_v3_norm_upprev_wer_filt.json"
#     "${input_root}/espnet/yodas/fr102/train_concatenated/train_espnet_yodas_manifest_zipped_whisper_large_v3_norm_upprev_wer_filt.json"
#     "${input_root}/espnet/yodas/fr103/train_concatenated/train_espnet_yodas_manifest_zipped_whisper_large_v3_norm_upprev_wer_filt.json"
# )
# validation_files=(
#     "${input_root}/mozilla-foundation/common_voice_17_0/fr/validation/validation_mozilla-foundation_common_voice_17_0_manifest.json"
#     "${input_root}/google/fleurs/fr_fr/validation/validation_google_fleurs_manifest.json"
# )

# de
train_files=(
    # ${input_root}/mozilla-foundation/common_voice_17_0/de/train_concatenated/train_mozilla-foundation_common_voice_17_0_manifest_whisper_large_v3_norm_wer_filt_wer_zipped_jz.json
    # ${input_root}/facebook/multilingual_librispeech/german/train_concatenated/train_facebook_multilingual_librispeech_manifest_whisper_large_v3_norm_upprev_wer_filt_zipped_jz.json
    # ${input_root}/facebook/voxpopuli/de/train_concatenated/train_facebook_voxpopuli_manifest_whisper_large_v3_norm_upprev_wer_filt_zipped_jz.json
    # ${input_root}/multilingual-tedx/de-de/train_concatenated/train_mtedx_asr_manifest_whisper_large_v3_norm_upprev_wer_filt_zipped_jz.json
    # ${input_root}/espnet/yodas/de000/train_concatenated/train_espnet_yodas_manifest_whisper_large_v3_norm_upprev_wer_filt_zipped_jz.json
    # ${input_root}/espnet/yodas/de100/train_concatenated/train_espnet_yodas_manifest_whisper_large_v3_norm_upprev_wer_filt_zipped_jz.json
    # ${input_root}/espnet/yodas/de101/train_concatenated/train_espnet_yodas_manifest_whisper_large_v3_norm_upprev_wer_filt_zipped_jz.json
    # ${input_root}/espnet/yodas/de102/train_concatenated/train_espnet_yodas_manifest_whisper_large_v3_norm_upprev_wer_filt_zipped_jz.json
    "${input_root}/final/train_manifest_de_wer20.json"
)
validation_files=(
    "${input_root}/mozilla-foundation/common_voice_17_0/de/validation/validation_mozilla-foundation_common_voice_17_0_manifest.json"
    "${input_root}/google/fleurs/de_de/validation/validation_google_fleurs_manifest.json"
)

# Use array expansion and join elements with a "+"
train_files=$(IFS=+; echo "${train_files[*]}")
validation_files=$(IFS=+; echo "${validation_files[*]}")

    # --max_train_samples 512 \
    # --max_eval_samples 512 \
    # --max_eval_samples 2000 \
    # --language "french" \
    # --ddp_timeout 7200 \
    # --gradient_checkpointing \
    # --overwrite_output_dir \

    # --attn_implementation "flash_attention_2" \
    # --attn_implementation "sdpa" \

    # --apply_audio_augmentation \
    # --background_noise_dir "/lustre/fsn1/projects/rech/gkb/commun/corpus/speech/musan" \
    # --audio_augmentation_prob 0.05 \

    # --timestamp_probability 0.5 \
    # --condition_on_prev_probability 0.5 \
    # --timestamp_probability 0.2 \
    # --condition_on_prev_probability 0.1 \

    # --mask_time_prob 0.1 \
    # --mask_time_length 10 \
    # --mask_time_min_masks 2 \
    # --mask_feature_prob 0.1 \
    # --mask_feature_length 10 \
    # --mask_feature_min_masks 2 \

    # --adam_beta1 0.9 \
    # --adam_beta2 0.95 \
    # --adam_epsilon "1e-5" \

    # --weight_decay "0.01" \
    # --weight_decay "0.1" \

    # --dropout 0.05 \
    # --dropout 0.1 \

    # --attention_dropout 0.05 \
    # --attention_dropout 0.1 \

    # --label_smoothing_factor 0.1 \

# export SCRIPT="run_distillation_b.py"
export SCRIPT="run_distillation_d.py"
# export SCRIPT="run_distillation_b_sf.py"

export SCRIPT_ARGS=" \
    --model_name_or_path $model_name_or_path \
    --teacher_model_name_or_path $teacher_model_name_or_path \
    --bpe_dropout 0.05 \
    --apply_spec_augment True \
    --mask_time_prob 0.22 \
    --mask_time_length 30 \
    --mask_time_min_masks 2 \
    --mask_feature_prob 0.22 \
    --mask_feature_length 14 \
    --mask_feature_min_masks 2 \
    --train_files $train_files \
    --validation_files $validation_files \
    --max_eval_samples 10000 \
    --audio_column_name audio_zip_filepath \
    --text_column_name whisper_transcript \
    --eval_text_column_name text \
    --duration_column_name duration \
    --max_duration_in_seconds 30 \
    --min_duration_in_seconds 1 \
    --prev_text_column_name prev_whisper_transcript \
    --condition_on_prev_column_name condition_on_prev \
    --language_column_name _language \
    --max_label_length 446 \
    --wer_threshold 20 \
    --timestamp_probability 0.5 \
    --condition_on_prev_probability 0.2 \
    --task transcribe \
    --preprocessing_num_workers $num_workers \
    --dataloader_num_workers 16 \
    --output_dir $output_dir \
    --num_train_epochs 160 \
    --per_device_train_batch_size 64 \
    --per_device_eval_batch_size 64 \
    --gradient_accumulation_steps 1 \
    --kl_weight 1.0 \
    --temperature 2.0 \
    --learning_rate 3e-4 \
    --warmup_ratio 0.03 \
    --lr_scheduler_type linear \
    --dtype bfloat16 \
    --gradient_checkpointing True \
    --attn_implementation sdpa \
    --freeze_encoder True \
    --freeze_embed_positions True \
    --logging_steps 10 \
    --eval_steps 1000 \
    --save_steps 1000 \
    --initial_save_and_eval_steps_multiplier 2 \
    --save_total_limit 10 \
    --predict_with_generate True \
    --generation_num_beams 1 \
    --return_timestamps False \
    --wandb_project $WANDB_PROJECT \
    --wandb_run_name $wandb_run_name \
    --do_train \
    --do_eval \
    "

export CMD="$LAUNCHER $SCRIPT $SCRIPT_ARGS"
srun $CMD

echo "END TIME: $(date)"
