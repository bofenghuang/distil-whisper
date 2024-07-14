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
export TOKENIZERS_PARALLELISM="false"
# export BITSANDBYTES_NOWELCOME="1"
# export HF_HUB_ENABLE_HF_TRANSFER="1"
export HF_HUB_OFFLINE="1"
export HF_DATASETS_OFFLINE="1"
export HF_EVALUATE_OFFLINE="1"

# CPUs
num_workers=320

# input_file="/projects/bhuang/corpus/speech/nemo_manifests/facebook/multilingual_librispeech/french/train/train_facebook_multilingual_librispeech_manifest.json"
# input_file="/projects/bhuang/corpus/speech/nemo_manifests/facebook/voxpopuli/fr/train/train_facebook_voxpopuli_manifest.json"
# input_file="/projects/bhuang/corpus/speech/nemo_manifests/espnet/yodas/fr000/train/train_espnet_yodas_manifest.json"

# input_file="/gpfsscratch/rech/cjc/commun/corpus/speech/nemo_manifests/facebook/multilingual_librispeech/italian/train/train_facebook_multilingual_librispeech_manifest.json"
# input_file="/gpfsscratch/rech/cjc/commun/corpus/speech/nemo_manifests/facebook/voxpopuli/it/train/train_facebook_voxpopuli_manifest.json"
# input_file="/gpfsscratch/rech/cjc/commun/corpus/speech/nemo_manifests/espnet/yodas/it000/train/train_espnet_yodas_manifest.json"
# input_file="/gpfsscratch/rech/cjc/commun/corpus/speech/nemo_manifests/espnet/yodas/it100/train/train_espnet_yodas_manifest.json"
# input_file="/gpfsscratch/rech/cjc/commun/corpus/speech/nemo_manifests/espnet/yodas/it101/train/train_espnet_yodas_manifest.json"

# take arg
input_file=$1

# output_file="${input_file%.*}_concatenated.json"
output_file="${input_file/\/train\//\/train_concatenated\/}"

python scripts/concat_asr_examples.py \
    --input_file_path $input_file \
    --output_file_path $output_file \
    --preprocessing_batch_size 1000 \
    --preprocessing_num_workers $num_workers

# tmp: split for parallel decoding
./scripts/split_file.sh "$output_file"

# python scripts/concat_asr_examples.py \
#     --input_file_path $output_file \
#     --output_file_path $output_file \
#     --preprocessing_batch_size 1000 \
#     --preprocessing_num_workers 320

# python scripts/tmp_add_ids.py \
#     --input_file_path_a $output_file \
#     --input_file_path_b ${output_file%.*}_whisper_large_v3_merged.json \
#     --output_file_path ${output_file%.*}_whisper_large_v3_merged.json \
#     --preprocessing_num_workers 320

echo "END TIME: $(date)"
