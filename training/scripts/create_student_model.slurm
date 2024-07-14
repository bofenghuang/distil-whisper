#!/bin/bash
# Copyright 2023  Bofeng Huang

#SBATCH --job-name=model-initing
#SBATCH --output=logs/%x/%j.out      # output file (%j = job ID)
#SBATCH --error=logs/%x/%j.err       # error file (%j = job ID)
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1          # crucial - only 1 task per dist per node!
#SBATCH --cpus-per-task=24           # number of cores per tasks
#SBATCH --gres=gpu:1                 # reserve 8 GPUs per node
#SBATCH --time=1:00:00               # maximum execution time (HH:MM:SS)
#SBATCH --qos=qos_gpu-dev            # QoS
#SBATCH --hint=nomultithread         # we get physical cores not logical
#SBATCH --partition=gpu_p2
#SBATCH --account=cjc@v100           # A100 accounting

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
# export OMP_NUM_THREADS="1"
# model_name_or_path="openai/whisper-large-v3"
# model_name_or_path="bofenghuang/whisper-large-v3-french"
model_name_or_path="/gpfswork/rech/cjc/commun/models/whisper/whisper-large-v3"

# tmp_model_id="$(echo "${model_name_or_path}" | sed -e "s/-/\_/g" -e "s/[ |=/]/-/g")"
# save_dir="./outputs/models/${tmp_model_id}_dec2_init"
# save_dir="/gpfswork/rech/cjc/commun/models/whisper/${tmp_model_id}_dec2_init"
save_dir="${model_name_or_path}_dec2_init"

python create_student_model.py \
    --teacher_checkpoint "$model_name_or_path" \
    --decoder_layers 2 \
    --save_dir "$save_dir"
