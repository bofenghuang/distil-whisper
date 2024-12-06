#!/bin/bash
#SBATCH --job-name=download
#SBATCH --output=logs/%x/%j.out                 # output file (%j = job ID)
#SBATCH --error=logs/%x/%j.err                  # error file (%j = job ID)
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1                     # crucial - only 1 task per dist per node!
#SBATCH --cpus-per-task=1                      # number of cores per tasks
#SBATCH --time=2:00:00                         # maximum execution time (HH:MM:SS)
#SBATCH --qos=qos_cpu-dev                        # QoS
#SBATCH --hint=nomultithread                    # we get physical cores not logical
#SBATCH --partition=prepost
#SBATCH --account=cjc@cpu

set -x -e

# set up environment
module purge
# module load arch/h100
module load git-lfs
module load unrar
module load anaconda-py3/2024.06
module load cuda/12.2.0
# conda activate speech
conda activate asr

# Convert datasets on Hugging Face Hub or local datasets to NeMo's manifest format.
# Also resample and export to wav files.

# https://github.com/pytorch/audio/issues/1021#issuecomment-726915239
# export OMP_NUM_THREADS="1"

# hf
# export HF_HOME="/projects/bhuang/.cache/huggingface"
# export REUSE_DATASET_IF_EXISTS="false"
# can't do multi worker with this on jz
# export HF_HUB_ENABLE_HF_TRANSFER="1"
# https://huggingface.co/docs/huggingface_hub/en/package_reference/environment_variables
# export HF_HUB_ETAG_TIMEOUT="60"
# export HF_HUB_DOWNLOAD_TIMEOUT="60"
# https://github.com/huggingface/transformers/issues/17611
# export CURL_CA_BUNDLE=""

num_proc="64"

    # --exclude original/*
    # --include "espnet/*" \

# huggingface-cli download \
#     --repo-type dataset \
#     --local-dir /lustre/fsn1/projects/rech/gkb/commun/corpus/speech/stt-pseudo-labeled-whisper-large-v3-multilingual \
#     --max-workers $num_proc \
#     bofenghuang/stt-pseudo-labeled-whisper-large-v3-multilingual

huggingface-cli upload-large-folder \
    --repo-type dataset \
    --exclude "*/valid*/*" "*/test*/*" "BrunoHays/*" "google/*" "eustlb/*" "gigant/*" "speech-recognition-community-v2/*" "code-switching/*" "final_*/*" \
    --num-workers $num_proc \
    multilingual-distil-whisper/stt-pseudo-labeled-whisper-large-v3-multilingual \
    /lustre/fsn1/projects/rech/gkb/commun/corpus/speech/stt-pseudo-labeled-whisper-large-v3-multilingual

echo "END TIME: $(date)"
