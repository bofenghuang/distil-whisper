#!/bin/bash
# Copyright 2024  Bofeng Huang

#SBATCH --job-name=stats
#SBATCH --output=logs/%x/%j.out      # output file (%j = job ID)
#SBATCH --error=logs/%x/%j.out       # error file (%j = job ID)
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1          # crucial - only 1 task per dist per node!
#SBATCH --cpus-per-task=16           # number of cores per tasks
#SBATCH --time 00:30:00              # maximum execution time (HH:MM:SS)
#SBATCH --qos=qos_cpu-dev            # QoS
#SBATCH --hint=nomultithread         # we get physical cores not logical
#SBATCH --partition=cpu_p1
#SBATCH --account=cjc@cpu            # cpu accounting

# get data duration by wer-filter

set -e

# set up environment
module purge
module load git-lfs
# module load unrar
module load anaconda-py3/2024.06
module load cuda/12.2.0
# conda activate speech
conda activate asr

echo "START TIME: $(date)"

# input_dir=/projects/bhuang/corpus/speech/nemo_manifests
input_dir=/lustre/fsn1/projects/rech/gkb/commun/corpus/speech/stt-pseudo-labeled-whisper-large-v3-multilingual

input_files=(
    # fr
    # $input_dir/mozilla-foundation/common_voice_17_0/fr/train/train_mozilla-foundation_common_voice_17_0_manifest_whisper_large_v3_norm_wer.json
    # $input_dir/facebook/multilingual_librispeech/french/train_concatenated/train_facebook_multilingual_librispeech_manifest_whisper_large_v3_norm_upprev_wer.json
    # $input_dir/facebook/voxpopuli/fr/train_concatenated/train_facebook_voxpopuli_manifest_whisper_large_v3_norm_upprev_wer.json
    # $input_dir/multilingual-tedx/fr-fr/train_concatenated/train_mtedx_asr_manifest_whisper_large_v3_norm_upprev_wer.json
    # $input_dir/espnet/yodas/fr000/train_concatenated/train_espnet_yodas_manifest_whisper_large_v3_norm_upprev_wer.json
    # $input_dir/espnet/yodas/fr100/train_concatenated/train_espnet_yodas_manifest_whisper_large_v3_norm_upprev_wer.json
    # $input_dir/espnet/yodas/fr101/train_concatenated/train_espnet_yodas_manifest_whisper_large_v3_norm_upprev_wer.json
    # $input_dir/espnet/yodas/fr102/train_concatenated/train_espnet_yodas_manifest_whisper_large_v3_norm_upprev_wer.json
    # $input_dir/espnet/yodas/fr103/train_concatenated/train_espnet_yodas_manifest_whisper_large_v3_norm_upprev_wer.json
    # # de
    # $input_dir/mozilla-foundation/common_voice_17_0/de/train/train_mozilla-foundation_common_voice_17_0_manifest_whisper_large_v3_norm_wer.json
    # $input_dir/facebook/multilingual_librispeech/german/train_concatenated/train_facebook_multilingual_librispeech_manifest_whisper_large_v3_norm_upprev_wer.json
    # $input_dir/facebook/voxpopuli/de/train_concatenated/train_facebook_voxpopuli_manifest_whisper_large_v3_norm_upprev_wer.json
    # $input_dir/multilingual-tedx/de-de/train_concatenated/train_mtedx_asr_manifest_whisper_large_v3_norm_upprev_wer.json
    # $input_dir/espnet/yodas/de000/train_concatenated/train_espnet_yodas_manifest_whisper_large_v3_norm_upprev_wer.json
    # $input_dir/espnet/yodas/de100/train_concatenated/train_espnet_yodas_manifest_whisper_large_v3_norm_upprev_wer.json
    # $input_dir/espnet/yodas/de101/train_concatenated/train_espnet_yodas_manifest_whisper_large_v3_norm_upprev_wer.json
    # $input_dir/espnet/yodas/de102/train_concatenated/train_espnet_yodas_manifest_whisper_large_v3_norm_upprev_wer.json
    # # es
    # $input_dir/mozilla-foundation/common_voice_17_0/es/train/train_mozilla-foundation_common_voice_17_0_manifest_whisper_large_v3_norm_wer.json
    # $input_dir/facebook/multilingual_librispeech/spanish/train_concatenated/train_facebook_multilingual_librispeech_manifest_whisper_large_v3_norm_upprev_wer.json
    # $input_dir/facebook/voxpopuli/es/train_concatenated/train_facebook_voxpopuli_manifest_whisper_large_v3_norm_upprev_wer.json
    # $input_dir/multilingual-tedx/es-es/train_concatenated/train_mtedx_asr_manifest_whisper_large_v3_norm_upprev_wer.json
    # $input_dir/espnet/yodas/es000/train_concatenated/train_espnet_yodas_manifest_whisper_large_v3_norm_upprev_wer.json
    # $input_dir/espnet/yodas/es100/train_concatenated/train_espnet_yodas_manifest_whisper_large_v3_norm_upprev_wer.json
    # $input_dir/espnet/yodas/es101/train_concatenated/train_espnet_yodas_manifest_whisper_large_v3_norm_upprev_wer.json
    # # it
    # $input_dir/mozilla-foundation/common_voice_17_0/it/train/train_mozilla-foundation_common_voice_17_0_manifest_whisper_large_v3_norm_wer.json
    # $input_dir/facebook/multilingual_librispeech/italian/train_concatenated/train_facebook_multilingual_librispeech_manifest_whisper_large_v3_norm_upprev_wer.json
    # $input_dir/facebook/voxpopuli/it/train_concatenated/train_facebook_voxpopuli_manifest_whisper_large_v3_norm_upprev_wer.json
    # $input_dir/multilingual-tedx/it-it/train_concatenated/train_mtedx_asr_manifest_whisper_large_v3_norm_upprev_wer.json
    # $input_dir/espnet/yodas/it000/train_concatenated/train_espnet_yodas_manifest_whisper_large_v3_norm_upprev_wer.json
    # $input_dir/espnet/yodas/it100/train_concatenated/train_espnet_yodas_manifest_whisper_large_v3_norm_upprev_wer.json
    # $input_dir/espnet/yodas/it101/train_concatenated/train_espnet_yodas_manifest_whisper_large_v3_norm_upprev_wer.json
    # # pt
    # $input_dir/mozilla-foundation/common_voice_17_0/pt/train/train_mozilla-foundation_common_voice_17_0_manifest_whisper_large_v3_norm_wer.json
    # $input_dir/facebook/multilingual_librispeech/portuguese/train_concatenated/train_facebook_multilingual_librispeech_manifest_whisper_large_v3_norm_upprev_wer.json
    # $input_dir/multilingual-tedx/pt-pt/train_concatenated/train_mtedx_asr_manifest_whisper_large_v3_norm_upprev_wer.json
    # $input_dir/espnet/yodas/pt000/train_concatenated/train_espnet_yodas_manifest_whisper_large_v3_norm_upprev_wer.json
    # $input_dir/espnet/yodas/pt100/train_concatenated/train_espnet_yodas_manifest_whisper_large_v3_norm_upprev_wer.json
    # $input_dir/espnet/yodas/pt101/train_concatenated/train_espnet_yodas_manifest_whisper_large_v3_norm_upprev_wer.json
    # $input_dir/espnet/yodas/pt102/train_concatenated/train_espnet_yodas_manifest_whisper_large_v3_norm_upprev_wer.json
    # # nl
    # $input_dir/mozilla-foundation/common_voice_17_0/nl/train/train_mozilla-foundation_common_voice_17_0_manifest_whisper_large_v3_norm_wer.json
    # $input_dir/facebook/multilingual_librispeech/dutch/train_concatenated/train_facebook_multilingual_librispeech_manifest_whisper_large_v3_norm_upprev_wer.json
    # $input_dir/facebook/voxpopuli/nl/train_concatenated/train_facebook_voxpopuli_manifest_whisper_large_v3_norm_upprev_wer.json
    # $input_dir/espnet/yodas/nl000/train_concatenated/train_espnet_yodas_manifest_whisper_large_v3_norm_upprev_wer.json
    # $input_dir/espnet/yodas/nl100/train_concatenated/train_espnet_yodas_manifest_whisper_large_v3_norm_upprev_wer.json
    # # en
    $input_dir/mozilla-foundation/common_voice_17_0/en/train/train_mozilla-foundation_common_voice_17_0_manifest_whisper_large_v3_norm_wer.json
    $input_dir/facebook/voxpopuli/en/train_concatenated/train_facebook_voxpopuli_manifest_whisper_large_v3_norm_upprev_wer.json
)


for input_file in "${input_files[@]}"; do
    python scripts/get_duration_stats.py $input_file
done

echo "END TIME: $(date)"
