#!/bin/bash
# Copyright 2023  Bofeng Huang

#SBATCH --job-name=stats
#SBATCH --output=logs/%x/%j.out      # output file (%j = job ID)
#SBATCH --error=logs/%x/%j.out       # error file (%j = job ID)
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1          # crucial - only 1 task per dist per node!
#SBATCH --cpus-per-task=4            # number of cores per tasks
#SBATCH --time 00:30:00              # maximum execution time (HH:MM:SS)
#SBATCH --qos=qos_cpu-dev             # QoS
#SBATCH --hint=nomultithread         # we get physical cores not logical
#SBATCH --partition=cpu_p1
#SBATCH --account=cjc@cpu            # cpu accounting

# set -x -e
set -e

# set up environment
module purge
module load git-lfs
module load unrar
module load anaconda-py3/2023.03
module load cuda/12.1.0
# conda activate speech
conda activate asr

# it
python scripts/get_duration_stats.py /gpfsscratch/rech/cjc/commun/corpus/speech/nemo_manifests/multilingual-tedx/it-it/train_concatenated/train_asr_whisper_large_v3_norm_upprev_wer_filt.json

# pt
# python scripts/get_duration_stats.py /gpfsscratch/rech/cjc/commun/corpus/speech/nemo_manifests/mozilla-foundation/common_voice_17_0/pt/train/train_mozilla-foundation_common_voice_17_0_manifest_whisper_large_v3_merged_norm_wer.json
# python scripts/get_duration_stats.py /gpfsscratch/rech/cjc/commun/corpus/speech/nemo_manifests/facebook/multilingual_librispeech/portuguese/train_concatenated/train_facebook_multilingual_librispeech_manifest_whisper_large_v3_merged_norm_upprev_wer_filt.json
python scripts/get_duration_stats.py /gpfsscratch/rech/cjc/commun/corpus/speech/nemo_manifests/multilingual-tedx/pt-pt/train_concatenated/train_asr_whisper_large_v3_norm_upprev_wer_filt.json
# python scripts/get_duration_stats.py /gpfsscratch/rech/cjc/commun/corpus/speech/nemo_manifests/espnet/yodas/pt000/train_concatenated/train_espnet_yodas_manifest_whisper_large_v3_norm_upprev_wer_filt.json

# de
# python scripts/get_duration_stats.py /gpfsscratch/rech/cjc/commun/corpus/speech/nemo_manifests/mozilla-foundation/common_voice_17_0/de/train/train_mozilla-foundation_common_voice_17_0_manifest_whisper_large_v3_norm_wer.json
# python scripts/get_duration_stats.py /gpfsscratch/rech/cjc/commun/corpus/speech/nemo_manifests/facebook/multilingual_librispeech/german/train_concatenated/train_facebook_multilingual_librispeech_manifest_whisper_large_v3_norm_upprev_wer_filt.json
# python scripts/get_duration_stats.py /gpfsscratch/rech/cjc/commun/corpus/speech/nemo_manifests/facebook/voxpopuli/de/train_concatenated/train_facebook_voxpopuli_manifest_whisper_large_v3_norm_upprev_wer_filt.json
python scripts/get_duration_stats.py /gpfsscratch/rech/cjc/commun/corpus/speech/nemo_manifests/multilingual-tedx/de-de/train_concatenated/train_asr_whisper_large_v3_norm_upprev_wer_filt.json
# python scripts/get_duration_stats.py /gpfsscratch/rech/cjc/commun/corpus/speech/nemo_manifests/espnet/yodas/de000/train_concatenated/train_espnet_yodas_manifest_whisper_large_v3_norm_upprev_wer_filt.json
