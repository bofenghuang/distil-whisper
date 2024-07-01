#!/bin/bash
# Copyright 2023  Bofeng Huang

#SBATCH --job-name=pseudo-labelling
#SBATCH --output=logs/%x/%j.out      # output file (%j = job ID)
#SBATCH --error=logs/%x/%j.err       # error file (%j = job ID)
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1          # crucial - only 1 task per dist per node!
#SBATCH --cpus-per-task=32           # number of cores per tasks
#SBATCH --gres=gpu:4                 # reserve 8 GPUs per node
#SBATCH --time=20:00:00              # maximum execution time (HH:MM:SS)
#SBATCH --qos=qos_gpu-t3             # QoS
#SBATCH --hint=nomultithread         # we get physical cores not logical
#SBATCH --constraint=a100
#SBATCH --account=cjc@a100           # A100 accounting

echo "START TIME: $(date)"

set -x -e

# set up environment
module purge
module load git-lfs
module load unrar
module load anaconda-py3/2023.03
module load cuda/12.1.0
# conda activate speech
conda activate asr

# export HF_HOME="/projects/bhuang/.cache/huggingface"
export OMP_NUM_THREADS="1"
# export CUDA_VISIBLE_DEVICES="4,5"

# Set your number of GPUs here
num_gpus=4

#   --dtype "bfloat16" \
# --max_label_length 128 \
    # --max_samples_per_split 1024 \

CMD="accelerate launch --multi_gpu --num_processes=$num_gpus"

model_name_or_path="openai/whisper-large-v3"
# model_name_or_path="bofenghuang/whisper-large-v3-french"

input_data_file="/projects/bhuang/corpus/speech/nemo_manifests/mozilla-foundation/common_voice_17_0/fr/train/train_mozilla-foundation_common_voice_17_0_manifest.json"

# tmp_model_id="$(echo "${model_name_or_path}" | sed -e "s/-/\_/g" -e "s/[ |=/]/-/g")"
# outdir="./outputs/data/$tmp_model_id"

tmp_model_id="$(echo "${model_name_or_path}" | sed -e "s/[ |=/-]/_/g")"
output_file="${input_data_file%.*}_${tmp_model_id}.json"

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

# python compute_wer.py \
#     --dataset_file "${outdir}/train-data.json" \
#     --final_output_json "${outdir}/train-data-wer.json" \
#     --text_column_name "text" \
#     --num_workers 64

echo "END TIME: $(date)"
