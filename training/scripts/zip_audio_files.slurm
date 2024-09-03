#!/bin/bash
# Copyright 2023  Bofeng Huang

#SBATCH --job-name=preprocessing
#SBATCH --output=logs/%x/%j.out      # output file (%j = job ID)
#SBATCH --error=logs/%x/%j.err       # error file (%j = job ID)
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1          # crucial - only 1 task per dist per node!
#SBATCH --cpus-per-task=16           # number of cores per tasks
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
# module load unrar
module load anaconda-py3/2024.06
module load cuda/12.2.0
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

# CPUs
num_workers=32

# take arg
# input_file=$1

# output_file="${input_file%.*}_zipped.json"

# python scripts/zip_audio_files.py \
#     --input_manifest_filepath $input_file \
#     --output_manifest_filepath $output_file \
#     --preprocessing_num_workers $num_workers

input_files=(
    /gpfsscratch/rech/cjc/commun/corpus/speech/nemo_manifests/mozilla-foundation/common_voice_17_0/it/train/train_mozilla-foundation_common_voice_17_0_manifest_whisper_large_v3_norm_wer_filt.json
    /gpfsscratch/rech/cjc/commun/corpus/speech/nemo_manifests/mozilla-foundation/common_voice_17_0/it/train_concatenated/train_mozilla-foundation_common_voice_17_0_manifest_whisper_large_v3_norm_wer_filt_wer.json
    /gpfsscratch/rech/cjc/commun/corpus/speech/nemo_manifests/mozilla-foundation/common_voice_17_0/it/train_concatenated_10/train_mozilla-foundation_common_voice_17_0_manifest_whisper_large_v3_norm_wer_filt10_wer.json
    /gpfsscratch/rech/cjc/commun/corpus/speech/nemo_manifests/facebook/multilingual_librispeech/italian/train_concatenated/train_facebook_multilingual_librispeech_manifest_whisper_large_v3_norm_upprev_wer_filt.json
    /gpfsscratch/rech/cjc/commun/corpus/speech/nemo_manifests/facebook/voxpopuli/it/train_concatenated/train_facebook_voxpopuli_manifest_whisper_large_v3_norm_upprev_wer_filt.json
    /gpfsscratch/rech/cjc/commun/corpus/speech/nemo_manifests/multilingual-tedx/it-it/train_concatenated/train_asr_whisper_large_v3_norm_upprev_wer_filt.json
    /gpfsscratch/rech/cjc/commun/corpus/speech/nemo_manifests/espnet/yodas/it000/train_concatenated/train_espnet_yodas_manifest_whisper_large_v3_merged_norm_upprev_wer_filt.json
    /gpfsscratch/rech/cjc/commun/corpus/speech/nemo_manifests/espnet/yodas/it100/train_concatenated/train_espnet_yodas_manifest_whisper_large_v3_merged_norm_upprev_wer_filt.json
    /gpfsscratch/rech/cjc/commun/corpus/speech/nemo_manifests/espnet/yodas/it101/train_concatenated/train_espnet_yodas_manifest_whisper_large_v3_merged_norm_upprev_wer_filt.json
    /gpfsscratch/rech/cjc/commun/corpus/speech/nemo_manifests/mozilla-foundation/common_voice_17_0/de/train/train_mozilla-foundation_common_voice_17_0_manifest_whisper_large_v3_norm_wer_filt.json
    /gpfsscratch/rech/cjc/commun/corpus/speech/nemo_manifests/mozilla-foundation/common_voice_17_0/de/train_concatenated/train_mozilla-foundation_common_voice_17_0_manifest_whisper_large_v3_norm_wer_filt_wer.json
    /gpfsscratch/rech/cjc/commun/corpus/speech/nemo_manifests/facebook/multilingual_librispeech/german/train_concatenated/train_facebook_multilingual_librispeech_manifest_whisper_large_v3_norm_upprev_wer_filt.json
    /gpfsscratch/rech/cjc/commun/corpus/speech/nemo_manifests/facebook/voxpopuli/de/train_concatenated/train_facebook_voxpopuli_manifest_whisper_large_v3_norm_upprev_wer_filt.json
    /gpfsscratch/rech/cjc/commun/corpus/speech/nemo_manifests/multilingual-tedx/de-de/train_concatenated/train_asr_whisper_large_v3_norm_upprev_wer_filt.json
    /gpfsscratch/rech/cjc/commun/corpus/speech/nemo_manifests/espnet/yodas/de000/train_concatenated/train_espnet_yodas_manifest_whisper_large_v3_norm_upprev_wer_filt.json
    /gpfsscratch/rech/cjc/commun/corpus/speech/nemo_manifests/espnet/yodas/de100/train_concatenated/train_espnet_yodas_manifest_whisper_large_v3_norm_upprev_wer_filt.json
    /gpfsscratch/rech/cjc/commun/corpus/speech/nemo_manifests/mozilla-foundation/common_voice_17_0/pt/train/train_mozilla-foundation_common_voice_17_0_manifest_whisper_large_v3_merged_norm_wer_filt.json
    /gpfsscratch/rech/cjc/commun/corpus/speech/nemo_manifests/mozilla-foundation/common_voice_17_0/pt/train_concatenated/train_mozilla-foundation_common_voice_17_0_manifest_whisper_large_v3_merged_norm_wer_filt_wer.json
    /gpfsscratch/rech/cjc/commun/corpus/speech/nemo_manifests/facebook/multilingual_librispeech/portuguese/train_concatenated/train_facebook_multilingual_librispeech_manifest_whisper_large_v3_merged_norm_upprev_wer_filt.json
    /gpfsscratch/rech/cjc/commun/corpus/speech/nemo_manifests/multilingual-tedx/pt-pt/train_concatenated/train_asr_whisper_large_v3_norm_upprev_wer_filt.json
    /gpfsscratch/rech/cjc/commun/corpus/speech/nemo_manifests/espnet/yodas/pt000/train_concatenated/train_espnet_yodas_manifest_whisper_large_v3_norm_upprev_wer_filt.json
    /gpfsscratch/rech/cjc/commun/corpus/speech/nemo_manifests/espnet/yodas/pt100/train_concatenated/train_espnet_yodas_manifest_whisper_large_v3_norm_upprev_wer_filt.json
)

for input_file in "${input_files[@]}"; do
    echo "$input_file"
    input_dir="${input_file%/*}"
    # output_file="${input_file%.*}_zipped.json"
    # python scripts/zip_audio_files.py --input_manifest_filepath $input_file --output_manifest_filepath $output_file --preprocessing_num_workers $num_workers
    # remove audios
    find $input_dir -mindepth 1 -type f -name "*.wav" -delete
    find $input_dir -mindepth 1 -type d -empty -delete
done


echo "END TIME: $(date)"
