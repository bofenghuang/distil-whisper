#!/bin/bash
# Copyright 2023  Bofeng Huang

#SBATCH --job-name=postprocessing
#SBATCH --output=logs/%x/%j.out      # output file (%j = job ID)
#SBATCH --error=logs/%x/%j.err       # error file (%j = job ID)
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1          # crucial - only 1 task per dist per node!
#SBATCH --cpus-per-task=40           # number of cores per tasks
#SBATCH --time 20:00:00              # maximum execution time (HH:MM:SS)
#SBATCH --qos=qos_cpu-t3             # QoS
#SBATCH --hint=nomultithread         # we get physical cores not logical
#SBATCH --partition=cpu_p1
#SBATCH --account=cjc@cpu            # cpu accounting

# Post-prcess pseudo-labelled data for mcv

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
num_workers=80

# input_file="/gpfsscratch/rech/cjc/commun/corpus/speech/nemo_manifests/mozilla-foundation/common_voice_17_0/it/train/train_mozilla-foundation_common_voice_17_0_manifest.json"

# take arg
input_file=$1

# tmp: merge splitted infered files
./scripts/merge_files.sh $input_file
input_file="${input_file%.*}_whisper_large_v3.json"

# normalize (timestamps)
python scripts/norm_whisper_transcript.py \
    --input_file_path "$input_file" \
    --output_file_path "${input_file%.*}_norm.json" \
    --num_workers $num_workers

# wer
python scripts/compute_wer.py \
    --input_file_path "${input_file%.*}_norm.json" \
    --output_file_path "${input_file%.*}_norm_wer.json" \
    --num_workers $num_workers

# filter (upper-case, wer)
python scripts/filter_whisper_transcript.py \
    --input_file_path "${input_file%.*}_norm_wer.json" \
    --output_file_path "${input_file%.*}_norm_wer_filt.json" \
    --wer_threshold 20 \
    --num_workers $num_workers

input_file="${input_file%.*}_norm_wer_filt.json"
output_file="${input_file/\/train\//\/train_concatenated\/}"
# output_file="${input_file/\/train\//\/train_concatenated_10\/}"

# concat
python scripts/concat_asr_examples.py \
    --input_file_path $input_file \
    --output_file_path $output_file \
    --preprocessing_batch_size 1000 \
    --preprocessing_num_workers $num_workers

# wer
python scripts/compute_wer.py \
    --input_file_path "$output_file" \
    --output_file_path "${output_file%.*}_wer.json" \
    --num_workers $num_workers

echo "END TIME: $(date)"
