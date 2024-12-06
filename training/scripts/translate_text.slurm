#!/bin/bash

#SBATCH --job-name=translate
#SBATCH --output=logs/%x/%j.out      # output file (%j = job ID)
#SBATCH --error=logs/%x/%j.err       # error file (%j = job ID)
#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH --ntasks-per-node=1          # crucial - only 1 task per dist per node!
#SBATCH --cpus-per-task=96           # number of cores per tasks
#SBATCH --gres=gpu:4                 # reserve number of GPUs per node
#SBATCH --time=20:00:00              # maximum execution time (HH:MM:SS)
#SBATCH --hint=nomultithread         # we get physical cores not logical
#SBATCH --constraint=h100
#SBATCH --account=gkb@h100           # account to use, with xyz the 3 letter code of your project

# translate whisper tanscript into English

set -x -e

echo "START TIME: $(date)"

# set up environment
module purge
module load arch/h100
module load git-lfs
# module load unrar
module load anaconda-py3/2024.06
module load cuda/12.2.0
# conda
conda activate llm

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

# slurm buffers python output by default, unbuffer python -u
export PYTHONUNBUFFERED="1"

# hf
# export HF_HOME="/projects/bhuang/.cache/huggingface"
export TOKENIZERS_PARALLELISM="false"
# export HF_HUB_ENABLE_HF_TRANSFER="1"
export HF_HUB_OFFLINE="1"
export HF_DATASETS_OFFLINE="1"
export HF_EVALUATE_OFFLINE="1"

# cuda
# export CUDA_VISIBLE_DEVICES="4,5,6,7"

# take arg
input_file=$1
# model_path=$2

# model
# model_path="/lustre/fswork/projects/rech/gkb/commun/models/pretrained/Qwen/Qwen2.5-72B-Instruct"
model_path="/lustre/fswork/projects/rech/gkb/commun/models/pretrained/meta-llama/Llama-3.1-70B-Instruct"
# model_path="/lustre/fswork/projects/rech/gkb/commun/models/pretrained/mistralai/Mistral-Large-Instruct-2411"

# tp_size=8
tp_size=4

output_file="${input_file%.*}_translated_new.json"

    # --max_samples 512 \

python scripts/gen_vllm.py \
    --prompt_file scripts/prompts/translate_text.txt \
    --input_file $input_file \
    --output_file $output_file \
    --id_column_name whisper_transcript \
    --instruct_column_name whisper_transcript \
    --output_column_name translated_whisper_transcript \
    --model_configs_file /lustre/fswork/projects/rech/gkb/uvl55hq/vgn/data/configs/model_configs.json \
    --model_name_or_path $model_path \
    --dtype bfloat16 \
    --tensor_parallel_size $tp_size \
    --gpu_memory_utilization 0.95 \
    --batch_size 1024 \
    --max_tokens 1024 \
    --max_model_len 1024 \
    --temperature 0 \
    --top_p 1.0

echo "END TIME: $(date)"
