#!/bin/bash
# Copyright 2023  Bofeng Huang

#SBATCH --job-name=pseudo-labelling-post
#SBATCH --output=logs/%x/%j.out      # output file (%j = job ID)
#SBATCH --error=logs/%x/%j.err       # error file (%j = job ID)
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1          # crucial - only 1 task per dist per node!
#SBATCH --cpus-per-task=40           # number of cores per tasks
#SBATCH --time 20:00:00              # maximum execution time (HH:MM:SS)
#SBATCH --qos=qos_gpu-t3             # QoS
#SBATCH --hint=nomultithread         # we get physical cores not logical
#SBATCH --partition=cpu_p1
#SBATCH --account=cjc@cpu            # cpu accounting

# Post-prcess pseudo-labelled data

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

# input_file="/gpfsscratch/rech/cjc/commun/corpus/speech/nemo_manifests/facebook/multilingual_librispeech/italian/train_concatenated/train_facebook_multilingual_librispeech_manifest_whisper_large_v3.json"
# input_file="/gpfsscratch/rech/cjc/commun/corpus/speech/nemo_manifests/facebook/voxpopuli/it/train_concatenated/train_facebook_voxpopuli_manifest_whisper_large_v3.json"
# input_file="/gpfsscratch/rech/cjc/commun/corpus/speech/nemo_manifests/espnet/yodas/it000/train_concatenated/train_espnet_yodas_manifest_whisper_large_v3_merged.json"
# input_file="/gpfsscratch/rech/cjc/commun/corpus/speech/nemo_manifests/espnet/yodas/it100/train_concatenated/train_espnet_yodas_manifest_whisper_large_v3_merged.json"
# input_file="/gpfsscratch/rech/cjc/commun/corpus/speech/nemo_manifests/espnet/yodas/it101/train_concatenated/train_espnet_yodas_manifest_whisper_large_v3_merged.json"

# take arg
input_file=$1

# normalize (timestamps)
python scripts/norm_whisper_transcript.py \
    --input_file_path "$input_file" \
    --output_file_path "${input_file%.*}_norm.json" \
    --num_workers $num_workers

# update prev_whisper_transcript
python scripts/update_prev_whisper_transcript.py \
    --input_file_path "${input_file%.*}_norm.json" \
    --output_file_path "${input_file%.*}_norm_upprev.json" \
    --num_workers $num_workers

# wer
python scripts/compute_wer.py \
    --input_file_path "${input_file%.*}_norm_upprev.json" \
    --output_file_path "${input_file%.*}_norm_upprev_wer.json" \
    --num_workers $num_workers

# filter (upper-case)
python scripts/filter_whisper_transcript.py \
    --input_file_path "${input_file%.*}_norm_upprev_wer.json" \
    --output_file_path "${input_file%.*}_norm_upprev_wer_filt.json" \
    --num_workers $num_workers

echo "END TIME: $(date)"
