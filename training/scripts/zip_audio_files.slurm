#!/bin/bash
# Copyright 2023  Bofeng Huang

#SBATCH --job-name=preprocessing
#SBATCH --output=logs/%x/%j.out      # output file (%j = job ID)
#SBATCH --error=logs/%x/%j.err       # error file (%j = job ID)
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1          # crucial - only 1 task per dist per node!
#SBATCH --cpus-per-task=40           # number of cores per tasks
#SBATCH --time 10:00:00              # maximum execution time (HH:MM:SS)
#SBATCH --qos=qos_cpu-t3             # QoS
#SBATCH --hint=nomultithread         # we get physical cores not logical
#SBATCH --partition=cpu_p1
#SBATCH --account=cjc@cpu            # cpu accounting

# Concat utterances

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

# https://github.com/pytorch/audio/issues/1021#issuecomment-726915239
# export OMP_NUM_THREADS="1"

# cuda
# export CUDA_VISIBLE_DEVICES=""

# hf
# export HF_HOME="/projects/bhuang/.cache/huggingface"
# export TOKENIZERS_PARALLELISM="false"
# export BITSANDBYTES_NOWELCOME="1"
# export HF_HUB_ENABLE_HF_TRANSFER="1"
export HF_HUB_OFFLINE="1"
export HF_DATASETS_OFFLINE="1"
export HF_EVALUATE_OFFLINE="1"

# take arg
input_file=$1

output_file="${input_file%.*}_zipped.json"

python scripts/zip_audio_files.py \
    --input_manifest_filepath $input_file \
    --output_manifest_filepath $output_file

echo "END TIME: $(date)"
