#!/bin/bash
# Copyright 2023  Bofeng Huang

#SBATCH --job-name=tmp-proc
#SBATCH --output=logs/%x/%j.out      # output file (%j = job ID)
#SBATCH --error=logs/%x/%j.err       # error file (%j = job ID)

##SBATCH --nodes=1
##SBATCH --exclusive
##SBATCH --ntasks-per-node=1          # crucial - only 1 task per dist per node!
##SBATCH --cpus-per-task=96           # number of cores per tasks
##SBATCH --gres=gpu:4                 # reserve number of GPUs per node
##SBATCH --time=20:00:00              # maximum execution time (HH:MM:SS)
##SBATCH --hint=nomultithread         # we get physical cores not logical
##SBATCH --constraint=h100
##SBATCH --account=gkb@h100           # account to use, with xyz the 3 letter code of your project

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1          # crucial - only 1 task per dist per node!
#SBATCH --cpus-per-task=40           # number of cores per tasks
#SBATCH --time 20:00:00              # maximum execution time (HH:MM:SS)
#SBATCH --qos=qos_cpu-t3             # QoS
#SBATCH --hint=nomultithread         # we get physical cores not logical
#SBATCH --partition=cpu_p1
#SBATCH --account=cjc@cpu            # cpu accounting

set -x -e

echo "START TIME: $(date)"

# set up environment
module purge
# module load arch/h100
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
# export HF_HUB_OFFLINE="1"
# export HF_DATASETS_OFFLINE="1"
# export HF_EVALUATE_OFFLINE="1"

# CPUs
# num_workers=64
num_workers=40

# compress data
####################################################################################################
# workdir="/gpfsscratch/rech/cjc/commun/corpus/speech"
# dirname="nemo_manifests"

# cd $workdir

# # tar -zcvf ${dirname}.tar.gz ${dirname}

# echo "wtf"
# tar -ztf ${dirname}.tar.gz  | grep test_asr_mcv13_manifest_normalized_pnc.json
# tar -zxvf ${dirname}.tar.gz ${dirname}/final/2023-11-28/test_asr_mcv13_manifest_normalized_pnc.json


# update audio path after migration
####################################################################################################
# input_dir="/lustre/fsn1/projects/rech/cjc/commun/corpus"
# input_dir="/lustre/fsn1/projects/rech/gkb/commun/corpus"

# find $input_dir -type f -name "*.json" -exec chmod g+w {} \;
# python scripts/tmp_update_path.py $input_dir 80

# cleanup, zip, remove audio files
####################################################################################################

# input_file=train_MLCommons_peoples_speech_manifest_zipped_whisper_large_v3.json

# input_dir="${input_file%/*}"

# for f in train_concatenated_*.zip; do
#     dirname=${f%.zip}
#     unzip "$f" -d "$dirname/"
# done

# python scripts/cleanup_audio_files.py \
#     --input_file_a "${input_file}" \
#     --input_file_b "${input_file%.*}_norm_upprev_wer_filt.json" \
#     --num_workers $num_workers

# python scripts/zip_audio_files.py \
#     --input_file_path "${input_file%.*}_norm_upprev_wer_filt.json" \
#     --output_file_path "${input_file%.*}_norm_upprev_wer_filt_zipped.json" \
#     --num_workers $num_workers

# # tmp: remove unzipped audio files
# find $input_dir -mindepth 1 -type f -name "*.wav" -delete
# find $input_dir -mindepth 1 -type d -empty -delete

# resplit zipped files bigger than 20GB for uploading
####################################################################################################

# cleanup_mls() {
#     local input_file=$1
    
#     input_dir="${input_file%/*}"
#     zipfile="${input_file%/*}.zip"

#     dirname="${zipfile%.zip}"
#     unzip "$zipfile" -d "$dirname/"
#     rm $zipfile

#     python scripts/split_folder.py \
#         --input_file_path "$input_file" \
#         --output_file_path "$input_file" \
#         --num_workers $num_workers

#     # zipping audio files
#     python scripts/zip_audio_files.py \
#         --input_file_path "$input_file" \
#         --output_file_path "${input_file%.*}_zipped.json" \
#         --num_workers $num_workers

#     # tmp: remove unzipped audio files
#     find $input_dir -mindepth 1 -type f -name "*.wav" -delete
#     find $input_dir -mindepth 1 -type d -empty -delete

# }

# cleanup_yodas() {
#     local input_file=$1

#     input_dir="${input_file%/*}"

#     for zipfile in $input_dir/*.zip; do
#         dirname="${zipfile%.zip}"
#         unzip "$zipfile" -d "$dirname/"
#         rm $zipfile
#     done

#     python scripts/split_folder.py \
#         --input_file_path "$input_file" \
#         --output_file_path "$input_file" \
#         --num_workers $num_workers

#     # zipping audio files
#     python scripts/zip_audio_files.py \
#         --input_file_path "$input_file" \
#         --output_file_path "${input_file%.*}_zipped.json" \
#         --num_workers $num_workers

#     # tmp: remove unzipped audio files
#     find $input_dir -mindepth 1 -type f -name "*.wav" -delete
#     find $input_dir -mindepth 1 -type d -empty -delete

# }

# # input_root=/projects/bhuang/corpus/speech/nemo_manifests
# input_root=/lustre/fsn1/projects/rech/gkb/commun/corpus/speech/stt-pseudo-labeled-whisper-large-v3-multilingual

# input_files=(
#     # $input_root/facebook/multilingual_librispeech/dutch/train_concatenated/train_facebook_multilingual_librispeech_manifest_whisper_large_v3_norm_upprev_wer_filt.json
#     # $input_root/facebook/multilingual_librispeech/french/train_concatenated/train_facebook_multilingual_librispeech_manifest_whisper_large_v3_norm_upprev_wer_filt.json
#     # $input_root/facebook/multilingual_librispeech/german/train_concatenated/train_facebook_multilingual_librispeech_manifest_whisper_large_v3_norm_upprev_wer_filt.json
#     # $input_root/facebook/multilingual_librispeech/italian/train_concatenated/train_facebook_multilingual_librispeech_manifest_whisper_large_v3_norm_upprev_wer_filt.json
#     # $input_root/facebook/multilingual_librispeech/portuguese/train_concatenated/train_facebook_multilingual_librispeech_manifest_whisper_large_v3_norm_upprev_wer_filt.json
#     # $input_root/facebook/multilingual_librispeech/spanish/train_concatenated/train_facebook_multilingual_librispeech_manifest_whisper_large_v3_norm_upprev_wer_filt.json
#     $input_root/MLCommons/peoples_speech/default/train_concatenated/train_MLCommons_peoples_speech_manifest_whisper_large_v3_norm_upprev_wer_filt.json
#     $input_root/openslr/librispeech_asr/train_concatenated/train_openslr_librispeech_asr_manifest_whisper_large_v3_norm_upprev_wer_filt.json
# )

# for input_file in "${input_files[@]}"; do
#     cleanup_mls "$input_file"
# done

# input_files=(
#     # $input_root/espnet/yodas/it100/train_concatenated/train_espnet_yodas_manifest_whisper_large_v3_norm_upprev_wer_filt.json
#     # $input_root/espnet/yodas/es000/train_concatenated/train_espnet_yodas_manifest_whisper_large_v3_norm_upprev_wer_filt.json
#     $input_root/LIUM/tedlium/release3/train_concatenated/train_LIUM_tedlium_manifest_whisper_large_v3_norm_upprev_wer_filt.json
# )

# for input_file in "${input_files[@]}"; do
#     cleanup_yodas "$input_file"
# done

####################################################################################################

# input_root=/projects/bhuang/corpus/speech/nemo_manifests
# input_root=/lustre/fsn1/projects/rech/gkb/commun/corpus/speech/stt-pseudo-labeled-whisper-large-v3-multilingual

# input_files=(
    # # it
    # $input_root/mozilla-foundation/common_voice_17_0/it/train_concatenated/train_mozilla-foundation_common_voice_17_0_manifest_whisper_large_v3_norm_wer_filt_wer_zipped.json
    # $input_root/facebook/multilingual_librispeech/italian/train_concatenated/train_facebook_multilingual_librispeech_manifest_whisper_large_v3_norm_upprev_wer_filt_zipped.json
    # $input_root/facebook/voxpopuli/it/train_concatenated/train_facebook_voxpopuli_manifest_whisper_large_v3_norm_upprev_wer_filt_zipped.json
    # $input_root/multilingual-tedx/it-it/train_concatenated/train_mtedx_asr_manifest_whisper_large_v3_norm_upprev_wer_filt_zipped.json
    # $input_root/espnet/yodas/it000/train_concatenated/train_espnet_yodas_manifest_whisper_large_v3_norm_upprev_wer_filt_zipped.json
    # $input_root/espnet/yodas/it100/train_concatenated/train_espnet_yodas_manifest_whisper_large_v3_norm_upprev_wer_filt_zipped.json
    # $input_root/espnet/yodas/it101/train_concatenated/train_espnet_yodas_manifest_whisper_large_v3_norm_upprev_wer_filt_zipped.json
    # # fr
    # $input_root/mozilla-foundation/common_voice_17_0/fr/train_concatenated/train_mozilla-foundation_common_voice_17_0_manifest_whisper_large_v3_norm_wer_filt_wer_zipped.json
    # $input_root/facebook/multilingual_librispeech/french/train_concatenated/train_facebook_multilingual_librispeech_manifest_whisper_large_v3_norm_upprev_wer_filt_zipped.json
    # $input_root/facebook/voxpopuli/fr/train_concatenated/train_facebook_voxpopuli_manifest_whisper_large_v3_norm_upprev_wer_filt_zipped.json
    # $input_root/multilingual-tedx/fr-fr/train_concatenated/train_mtedx_asr_manifest_whisper_large_v3_norm_upprev_wer_filt_zipped.json
    # $input_root/espnet/yodas/fr000/train_concatenated/train_espnet_yodas_manifest_whisper_large_v3_norm_upprev_wer_filt_zipped.json
    # $input_root/espnet/yodas/fr100/train_concatenated/train_espnet_yodas_manifest_whisper_large_v3_norm_upprev_wer_filt_zipped.json
    # $input_root/espnet/yodas/fr101/train_concatenated/train_espnet_yodas_manifest_whisper_large_v3_norm_upprev_wer_filt_zipped.json
    # $input_root/espnet/yodas/fr102/train_concatenated/train_espnet_yodas_manifest_whisper_large_v3_norm_upprev_wer_filt_zipped.json
    # $input_root/espnet/yodas/fr103/train_concatenated/train_espnet_yodas_manifest_whisper_large_v3_norm_upprev_wer_filt_zipped.json
    # # es
    # $input_root/mozilla-foundation/common_voice_17_0/es/train_concatenated/train_mozilla-foundation_common_voice_17_0_manifest_whisper_large_v3_norm_wer_filt_wer_zipped.json
    # $input_root/facebook/multilingual_librispeech/spanish/train_concatenated/train_facebook_multilingual_librispeech_manifest_whisper_large_v3_norm_upprev_wer_filt_zipped.json
    # $input_root/facebook/voxpopuli/es/train_concatenated/train_facebook_voxpopuli_manifest_whisper_large_v3_norm_upprev_wer_filt_zipped.json
    # $input_root/multilingual-tedx/es-es/train_concatenated/train_mtedx_asr_manifest_whisper_large_v3_norm_upprev_wer_filt_zipped.json
    # $input_root/espnet/yodas/es000/train_concatenated/train_espnet_yodas_manifest_whisper_large_v3_norm_upprev_wer_filt_zipped.json
    # $input_root/espnet/yodas/es100/train_concatenated/train_espnet_yodas_manifest_whisper_large_v3_norm_upprev_wer_filt_zipped.json
    # $input_root/espnet/yodas/es101/train_concatenated/train_espnet_yodas_manifest_whisper_large_v3_norm_upprev_wer_filt_zipped.json
    # # pt
    # $input_root/mozilla-foundation/common_voice_17_0/pt/train_concatenated/train_mozilla-foundation_common_voice_17_0_manifest_whisper_large_v3_norm_wer_filt_wer_zipped.json
    # $input_root/facebook/multilingual_librispeech/portuguese/train_concatenated/train_facebook_multilingual_librispeech_manifest_whisper_large_v3_norm_upprev_wer_filt_zipped.json
    # $input_root/multilingual-tedx/pt-pt/train_concatenated/train_mtedx_asr_manifest_whisper_large_v3_norm_upprev_wer_filt_zipped.json
    # $input_root/espnet/yodas/pt000/train_concatenated/train_espnet_yodas_manifest_whisper_large_v3_norm_upprev_wer_filt_zipped.json
    # $input_root/espnet/yodas/pt100/train_concatenated/train_espnet_yodas_manifest_whisper_large_v3_norm_upprev_wer_filt_zipped.json
    # $input_root/espnet/yodas/pt101/train_concatenated/train_espnet_yodas_manifest_whisper_large_v3_norm_upprev_wer_filt_zipped.json
    # $input_root/espnet/yodas/pt102/train_concatenated/train_espnet_yodas_manifest_whisper_large_v3_norm_upprev_wer_filt_zipped.json
    # $input_root/espnet/yodas/pt103/train_concatenated/train_espnet_yodas_manifest_whisper_large_v3_norm_upprev_wer_filt_zipped.json
    # # de
    # $input_root/mozilla-foundation/common_voice_17_0/de/train_concatenated/train_mozilla-foundation_common_voice_17_0_manifest_whisper_large_v3_norm_wer_filt_wer_zipped.json
    # $input_root/facebook/multilingual_librispeech/german/train_concatenated/train_facebook_multilingual_librispeech_manifest_whisper_large_v3_norm_upprev_wer_filt_zipped.json
    # $input_root/facebook/voxpopuli/de/train_concatenated/train_facebook_voxpopuli_manifest_whisper_large_v3_norm_upprev_wer_filt_zipped.json
    # $input_root/multilingual-tedx/de-de/train_concatenated/train_mtedx_asr_manifest_whisper_large_v3_norm_upprev_wer_filt_zipped.json
    # $input_root/espnet/yodas/de000/train_concatenated/train_espnet_yodas_manifest_whisper_large_v3_norm_upprev_wer_filt_zipped.json
    # $input_root/espnet/yodas/de100/train_concatenated/train_espnet_yodas_manifest_whisper_large_v3_norm_upprev_wer_filt_zipped.json
    # $input_root/espnet/yodas/de101/train_concatenated/train_espnet_yodas_manifest_whisper_large_v3_norm_upprev_wer_filt_zipped.json
    # $input_root/espnet/yodas/de102/train_concatenated/train_espnet_yodas_manifest_whisper_large_v3_norm_upprev_wer_filt_zipped.json
    # # nl
    # $input_root/mozilla-foundation/common_voice_17_0/nl/train_concatenated/train_mozilla-foundation_common_voice_17_0_manifest_whisper_large_v3_norm_wer_filt_wer_zipped.json
    # $input_root/facebook/multilingual_librispeech/dutch/train_concatenated/train_facebook_multilingual_librispeech_manifest_whisper_large_v3_norm_upprev_wer_filt_zipped.json
    # $input_root/facebook/voxpopuli/nl/train_concatenated/train_facebook_voxpopuli_manifest_whisper_large_v3_norm_upprev_wer_filt_zipped.json
    # $input_root/espnet/yodas/nl000/train_concatenated/train_espnet_yodas_manifest_whisper_large_v3_norm_upprev_wer_filt_zipped.json
    # $input_root/espnet/yodas/nl100/train_concatenated/train_espnet_yodas_manifest_whisper_large_v3_norm_upprev_wer_filt_zipped.json
    # en
    # $input_root/mozilla-foundation/common_voice_17_0/en/train_concatenated/train_mozilla-foundation_common_voice_17_0_manifest_whisper_large_v3_norm_wer_filt_wer_zipped.json
    # $input_root/openslr/librispeech_asr/train_concatenated/train_openslr_librispeech_asr_manifest_whisper_large_v3_norm_upprev_wer_filt_zipped.json
    # $input_root/facebook/voxpopuli/en/train_concatenated/train_facebook_voxpopuli_manifest_whisper_large_v3_norm_upprev_wer_filt_zipped.json
    # $input_root/LIUM/tedlium/release3/train_concatenated/train_LIUM_tedlium_manifest_whisper_large_v3_norm_upprev_wer_filt_zipped.json
    # $input_root/MLCommons/peoples_speech/default/train_concatenated/train_MLCommons_peoples_speech_manifest_whisper_large_v3_norm_upprev_wer_filt_zipped.json
    # $input_root/speechcolab/gigaspeech/l/train_concatenated/train_speechcolab_gigaspeech_manifest_whisper_large_v3_norm_upprev_wer_filt_zipped.json
    # $input_root/distil-whisper/ami-ihm/ihm/train_concatenated/train_distil-whisper_ami-ihm_manifest_whisper_large_v3_norm_upprev_wer_filt_zipped.json
    # $input_root/distil-whisper/ami-sdm/sdm/train_concatenated/train_distil-whisper_ami-sdm_manifest_whisper_large_v3_norm_upprev_wer_filt_zipped.json
    # $input_root/espnet/yodas/en000/train_concatenated/train_espnet_yodas_manifest_whisper_large_v3_norm_upprev_wer_filt_zipped.json
    # $input_root/espnet/yodas/en001/train_concatenated/train_espnet_yodas_manifest_whisper_large_v3_norm_upprev_wer_filt_zipped.json
    # $input_root/espnet/yodas/en002/train_concatenated/train_espnet_yodas_manifest_whisper_large_v3_norm_upprev_wer_filt_zipped.json
    # $input_root/espnet/yodas/en003/train_concatenated/train_espnet_yodas_manifest_whisper_large_v3_norm_upprev_wer_filt_zipped.json
    # $input_root/espnet/yodas/en004/train_concatenated/train_espnet_yodas_manifest_whisper_large_v3_norm_upprev_wer_filt_zipped.json
    # $input_root/espnet/yodas/en005/train_concatenated/train_espnet_yodas_manifest_whisper_large_v3_norm_upprev_wer_filt_zipped.json
    # $input_root/espnet/yodas/en100/train_concatenated/train_espnet_yodas_manifest_whisper_large_v3_norm_upprev_wer_filt_zipped.json
    # $input_root/espnet/yodas/en101/train_concatenated/train_espnet_yodas_manifest_whisper_large_v3_norm_upprev_wer_filt_zipped.json
    # $input_root/espnet/yodas/en102/train_concatenated/train_espnet_yodas_manifest_whisper_large_v3_norm_upprev_wer_filt_zipped.json
    # $input_root/espnet/yodas/en103/train_concatenated/train_espnet_yodas_manifest_whisper_large_v3_norm_upprev_wer_filt_zipped.json
    # $input_root/espnet/yodas/en104/train_concatenated/train_espnet_yodas_manifest_whisper_large_v3_norm_upprev_wer_filt_zipped.json
    # $input_root/espnet/yodas/en105/train_concatenated/train_espnet_yodas_manifest_whisper_large_v3_norm_upprev_wer_filt_zipped.json
    # $input_root/espnet/yodas/en106/train_concatenated/train_espnet_yodas_manifest_whisper_large_v3_norm_upprev_wer_filt_zipped.json
    # $input_root/espnet/yodas/en107/train_concatenated/train_espnet_yodas_manifest_whisper_large_v3_norm_upprev_wer_filt_zipped.json
    # # $input_root/espnet/yodas/en108/train_concatenated/train_espnet_yodas_manifest_whisper_large_v3_norm_upprev_wer_filt_zipped.json
    # $input_root/espnet/yodas/en109/train_concatenated/train_espnet_yodas_manifest_whisper_large_v3_norm_upprev_wer_filt_zipped.json
    # $input_root/espnet/yodas/en110/train_concatenated/train_espnet_yodas_manifest_whisper_large_v3_norm_upprev_wer_filt_zipped.json
    # $input_root/espnet/yodas/en111/train_concatenated/train_espnet_yodas_manifest_whisper_large_v3_norm_upprev_wer_filt_zipped.json
    # $input_root/espnet/yodas/en112/train_concatenated/train_espnet_yodas_manifest_whisper_large_v3_norm_upprev_wer_filt_zipped.json
    # $input_root/espnet/yodas/en113/train_concatenated/train_espnet_yodas_manifest_whisper_large_v3_norm_upprev_wer_filt_zipped.json
    # $input_root/espnet/yodas/en114/train_concatenated/train_espnet_yodas_manifest_whisper_large_v3_norm_upprev_wer_filt_zipped.json
    # $input_root/espnet/yodas/en115/train_concatenated/train_espnet_yodas_manifest_whisper_large_v3_norm_upprev_wer_filt_zipped.json
    # $input_root/espnet/yodas/en116/train_concatenated/train_espnet_yodas_manifest_whisper_large_v3_norm_upprev_wer_filt_zipped.json
    # $input_root/espnet/yodas/en117/train_concatenated/train_espnet_yodas_manifest_whisper_large_v3_norm_upprev_wer_filt_zipped.json
    # $input_root/espnet/yodas/en118/train_concatenated/train_espnet_yodas_manifest_whisper_large_v3_norm_upprev_wer_filt_zipped.json
    # $input_root/espnet/yodas/en119/train_concatenated/train_espnet_yodas_manifest_whisper_large_v3_norm_upprev_wer_filt_zipped.json
    # $input_root/espnet/yodas/en120/train_concatenated/train_espnet_yodas_manifest_whisper_large_v3_norm_upprev_wer_filt_zipped.json
    # $input_root/espnet/yodas/en121/train_concatenated/train_espnet_yodas_manifest_whisper_large_v3_norm_upprev_wer_filt_zipped.json
    # $input_root/espnet/yodas/en122/train_concatenated/train_espnet_yodas_manifest_whisper_large_v3_norm_upprev_wer_filt_zipped.json
    # $input_root/espnet/yodas/en123/train_concatenated/train_espnet_yodas_manifest_whisper_large_v3_norm_upprev_wer_filt_zipped.json
    # $input_root/espnet/yodas/en124/train_concatenated/train_espnet_yodas_manifest_whisper_large_v3_norm_upprev_wer_filt_zipped.json
    # $input_root/espnet/yodas/en125/train_concatenated/train_espnet_yodas_manifest_whisper_large_v3_norm_upprev_wer_filt_zipped.json
    # $input_root/espnet/yodas/en126/train_concatenated/train_espnet_yodas_manifest_whisper_large_v3_norm_upprev_wer_filt_zipped.json
    # $input_root/espnet/yodas/en127/train_concatenated/train_espnet_yodas_manifest_whisper_large_v3_norm_upprev_wer_filt_zipped.json
#     # validation
#     $input_root/mozilla-foundation/common_voice_17_0/it/validation/validation_mozilla-foundation_common_voice_17_0_manifest.json
#     $input_root/mozilla-foundation/common_voice_17_0/fr/validation/validation_mozilla-foundation_common_voice_17_0_manifest.json
#     $input_root/mozilla-foundation/common_voice_17_0/es/validation/validation_mozilla-foundation_common_voice_17_0_manifest.json
#     $input_root/mozilla-foundation/common_voice_17_0/pt/validation/validation_mozilla-foundation_common_voice_17_0_manifest.json
#     $input_root/mozilla-foundation/common_voice_17_0/de/validation/validation_mozilla-foundation_common_voice_17_0_manifest.json
#     $input_root/mozilla-foundation/common_voice_17_0/nl/validation/validation_mozilla-foundation_common_voice_17_0_manifest.json
#     $input_root/mozilla-foundation/common_voice_17_0/en/validation/validation_mozilla-foundation_common_voice_17_0_manifest.json
#     $input_root/google/fleurs/it_it/validation/validation_google_fleurs_manifest.json
#     $input_root/google/fleurs/fr_fr/validation/validation_google_fleurs_manifest.json
#     $input_root/google/fleurs/es_419/validation/validation_google_fleurs_manifest.json
#     $input_root/google/fleurs/pt_br/validation/validation_google_fleurs_manifest.json
#     $input_root/google/fleurs/de_de/validation/validation_google_fleurs_manifest.json
#     $input_root/google/fleurs/nl_nl/validation/validation_google_fleurs_manifest.json
#     $input_root/google/fleurs/en_us/validation/validation_google_fleurs_manifest.json
# )

# for input_file in "${input_files[@]}"; do

#     # [ -f $input_file ] || echo $input_file

#     # zip and remove wav
#     ####################################################################################################

#     # python scripts/zip_audio_files.py \
#     #     --input_file_path "$input_file" \
#     #     --output_file_path "${input_file%.*}_zipped.json" \
#     #     --num_workers 128

#     # # tmp: remove unzipped audio files
#     # input_dir=${input_file%/*}    
#     # find $input_dir -mindepth 1 -type f -name "*.wav" -delete
#     # find $input_dir -mindepth 1 -type d -empty -delete

#     # # tmp: remove splitted_files
#     # tmp_dir="$input_dir/splitted_files"
#     # [ -d "$tmp_dir" ] && rm -r "$tmp_dir"

#     # check presence
#     ####################################################################################################
#     # input_file="${input_file%.*}_zipped.json"
#     # python scripts/validate_audio_files_presence.py $input_file audio_zip_filepath

#     # update migrated path
#     ####################################################################################################
#     # python scripts/tmp_update_path.py $input_file ${input_file%.*}_jz.json $num_workers
#     # python scripts/validate_audio_files_presence.py ${input_file%.*}_jz.json audio_zip_filepath || exit 1
# done

# add lang col
####################################################################################################

# input_files=(
#     $input_root/mozilla-foundation/common_voice_17_0/it/validation/validation_mozilla-foundation_common_voice_17_0_manifest.json
#     $input_root/google/fleurs/it_it/validation/validation_google_fleurs_manifest.json
# )
# lang=it
# for input_file in "${input_files[@]}"; do
#     python scripts/insert_column.py $input_file $input_file _language $lang
# done

# input_files=(
#     $input_root/mozilla-foundation/common_voice_17_0/fr/validation/validation_mozilla-foundation_common_voice_17_0_manifest.json
#     $input_root/google/fleurs/fr_fr/validation/validation_google_fleurs_manifest.json

# )
# lang=fr
# for input_file in "${input_files[@]}"; do
#     python scripts/insert_column.py $input_file $input_file _language $lang
# done

# input_files=(
#     $input_root/mozilla-foundation/common_voice_17_0/es/validation/validation_mozilla-foundation_common_voice_17_0_manifest.json
#     $input_root/google/fleurs/es_419/validation/validation_google_fleurs_manifest.json

# )
# lang=es
# for input_file in "${input_files[@]}"; do
#     python scripts/insert_column.py $input_file $input_file _language $lang
# done

# input_files=(
#     $input_root/mozilla-foundation/common_voice_17_0/pt/validation/validation_mozilla-foundation_common_voice_17_0_manifest.json
#     $input_root/google/fleurs/pt_br/validation/validation_google_fleurs_manifest.json

# )
# lang=pt
# for input_file in "${input_files[@]}"; do
#     python scripts/insert_column.py $input_file $input_file _language $lang
# done

# input_files=(
#     $input_root/mozilla-foundation/common_voice_17_0/de/validation/validation_mozilla-foundation_common_voice_17_0_manifest.json
#     $input_root/google/fleurs/de_de/validation/validation_google_fleurs_manifest.json

# )
# lang=de
# for input_file in "${input_files[@]}"; do
#     python scripts/insert_column.py $input_file $input_file _language $lang
# done

# input_files=(
#     $input_root/mozilla-foundation/common_voice_17_0/nl/validation/validation_mozilla-foundation_common_voice_17_0_manifest.json
#     $input_root/google/fleurs/nl_nl/validation/validation_google_fleurs_manifest.json

# )
# lang=nl
# for input_file in "${input_files[@]}"; do
#     python scripts/insert_column.py $input_file $input_file _language $lang
# done

# input_files=(
#     $input_root/mozilla-foundation/common_voice_17_0/en/validation/validation_mozilla-foundation_common_voice_17_0_manifest.json
#     $input_root/google/fleurs/en_us/validation/validation_google_fleurs_manifest.json

# )
# lang=en
# for input_file in "${input_files[@]}"; do
#     python scripts/insert_column.py $input_file $input_file _language $lang
# done

####################################################################################################

# input_root=/lustre/fsn1/projects/rech/gkb/commun/corpus/speech/stt-pseudo-labeled-whisper-large-v3-multilingual

# input_files=(
#     # it
#     $input_root/mozilla-foundation/common_voice_17_0/it/train_concatenated/train_mozilla-foundation_common_voice_17_0_manifest_whisper_large_v3_norm_wer_filt_wer_zipped_jz.json
#     $input_root/facebook/multilingual_librispeech/italian/train_concatenated/train_facebook_multilingual_librispeech_manifest_whisper_large_v3_norm_upprev_wer_filt_zipped_jz.json
#     $input_root/facebook/voxpopuli/it/train_concatenated/train_facebook_voxpopuli_manifest_whisper_large_v3_norm_upprev_wer_filt_zipped_jz.json
#     $input_root/multilingual-tedx/it-it/train_concatenated/train_mtedx_asr_manifest_whisper_large_v3_norm_upprev_wer_filt_zipped_jz.json
#     $input_root/espnet/yodas/it000/train_concatenated/train_espnet_yodas_manifest_whisper_large_v3_norm_upprev_wer_filt_zipped_jz.json
#     $input_root/espnet/yodas/it100/train_concatenated/train_espnet_yodas_manifest_whisper_large_v3_norm_upprev_wer_filt_zipped_jz.json
#     $input_root/espnet/yodas/it101/train_concatenated/train_espnet_yodas_manifest_whisper_large_v3_norm_upprev_wer_filt_zipped_jz.json
#     # fr
#     $input_root/mozilla-foundation/common_voice_17_0/fr/train_concatenated/train_mozilla-foundation_common_voice_17_0_manifest_whisper_large_v3_norm_wer_filt_wer_zipped_jz.json
#     $input_root/facebook/multilingual_librispeech/french/train_concatenated/train_facebook_multilingual_librispeech_manifest_whisper_large_v3_norm_upprev_wer_filt_zipped_jz.json
#     $input_root/facebook/voxpopuli/fr/train_concatenated/train_facebook_voxpopuli_manifest_whisper_large_v3_norm_upprev_wer_filt_zipped_jz.json
#     $input_root/multilingual-tedx/fr-fr/train_concatenated/train_mtedx_asr_manifest_whisper_large_v3_norm_upprev_wer_filt_zipped_jz.json
#     $input_root/espnet/yodas/fr000/train_concatenated/train_espnet_yodas_manifest_whisper_large_v3_norm_upprev_wer_filt_zipped_jz.json
#     $input_root/espnet/yodas/fr100/train_concatenated/train_espnet_yodas_manifest_whisper_large_v3_norm_upprev_wer_filt_zipped_jz.json
#     $input_root/espnet/yodas/fr101/train_concatenated/train_espnet_yodas_manifest_whisper_large_v3_norm_upprev_wer_filt_zipped_jz.json
#     $input_root/espnet/yodas/fr102/train_concatenated/train_espnet_yodas_manifest_whisper_large_v3_norm_upprev_wer_filt_zipped_jz.json
#     $input_root/espnet/yodas/fr103/train_concatenated/train_espnet_yodas_manifest_whisper_large_v3_norm_upprev_wer_filt_zipped_jz.json
#     # es
#     $input_root/mozilla-foundation/common_voice_17_0/es/train_concatenated/train_mozilla-foundation_common_voice_17_0_manifest_whisper_large_v3_norm_wer_filt_wer_zipped_jz.json
#     $input_root/facebook/multilingual_librispeech/spanish/train_concatenated/train_facebook_multilingual_librispeech_manifest_whisper_large_v3_norm_upprev_wer_filt_zipped_jz.json
#     $input_root/facebook/voxpopuli/es/train_concatenated/train_facebook_voxpopuli_manifest_whisper_large_v3_norm_upprev_wer_filt_zipped_jz.json
#     $input_root/multilingual-tedx/es-es/train_concatenated/train_mtedx_asr_manifest_whisper_large_v3_norm_upprev_wer_filt_zipped_jz.json
#     $input_root/espnet/yodas/es000/train_concatenated/train_espnet_yodas_manifest_whisper_large_v3_norm_upprev_wer_filt_zipped_jz.json
#     $input_root/espnet/yodas/es100/train_concatenated/train_espnet_yodas_manifest_whisper_large_v3_norm_upprev_wer_filt_zipped_jz.json
#     $input_root/espnet/yodas/es101/train_concatenated/train_espnet_yodas_manifest_whisper_large_v3_norm_upprev_wer_filt_zipped_jz.json
#     # pt
#     $input_root/mozilla-foundation/common_voice_17_0/pt/train_concatenated/train_mozilla-foundation_common_voice_17_0_manifest_whisper_large_v3_norm_wer_filt_wer_zipped_jz.json
#     $input_root/facebook/multilingual_librispeech/portuguese/train_concatenated/train_facebook_multilingual_librispeech_manifest_whisper_large_v3_norm_upprev_wer_filt_zipped_jz.json
#     $input_root/multilingual-tedx/pt-pt/train_concatenated/train_mtedx_asr_manifest_whisper_large_v3_norm_upprev_wer_filt_zipped_jz.json
#     $input_root/espnet/yodas/pt000/train_concatenated/train_espnet_yodas_manifest_whisper_large_v3_norm_upprev_wer_filt_zipped_jz.json
#     $input_root/espnet/yodas/pt100/train_concatenated/train_espnet_yodas_manifest_whisper_large_v3_norm_upprev_wer_filt_zipped_jz.json
#     $input_root/espnet/yodas/pt101/train_concatenated/train_espnet_yodas_manifest_whisper_large_v3_norm_upprev_wer_filt_zipped_jz.json
#     $input_root/espnet/yodas/pt102/train_concatenated/train_espnet_yodas_manifest_whisper_large_v3_norm_upprev_wer_filt_zipped_jz.json
#     $input_root/espnet/yodas/pt103/train_concatenated/train_espnet_yodas_manifest_whisper_large_v3_norm_upprev_wer_filt_zipped_jz.json
#     # de
#     $input_root/mozilla-foundation/common_voice_17_0/de/train_concatenated/train_mozilla-foundation_common_voice_17_0_manifest_whisper_large_v3_norm_wer_filt_wer_zipped_jz.json
#     $input_root/facebook/multilingual_librispeech/german/train_concatenated/train_facebook_multilingual_librispeech_manifest_whisper_large_v3_norm_upprev_wer_filt_zipped_jz.json
#     $input_root/facebook/voxpopuli/de/train_concatenated/train_facebook_voxpopuli_manifest_whisper_large_v3_norm_upprev_wer_filt_zipped_jz.json
#     $input_root/multilingual-tedx/de-de/train_concatenated/train_mtedx_asr_manifest_whisper_large_v3_norm_upprev_wer_filt_zipped_jz.json
#     $input_root/espnet/yodas/de000/train_concatenated/train_espnet_yodas_manifest_whisper_large_v3_norm_upprev_wer_filt_zipped_jz.json
#     $input_root/espnet/yodas/de100/train_concatenated/train_espnet_yodas_manifest_whisper_large_v3_norm_upprev_wer_filt_zipped_jz.json
#     $input_root/espnet/yodas/de101/train_concatenated/train_espnet_yodas_manifest_whisper_large_v3_norm_upprev_wer_filt_zipped_jz.json
#     $input_root/espnet/yodas/de102/train_concatenated/train_espnet_yodas_manifest_whisper_large_v3_norm_upprev_wer_filt_zipped_jz.json
#     # nl
#     $input_root/mozilla-foundation/common_voice_17_0/nl/train_concatenated/train_mozilla-foundation_common_voice_17_0_manifest_whisper_large_v3_norm_wer_filt_wer_zipped_jz.json
#     $input_root/facebook/multilingual_librispeech/dutch/train_concatenated/train_facebook_multilingual_librispeech_manifest_whisper_large_v3_norm_upprev_wer_filt_zipped_jz.json
#     $input_root/facebook/voxpopuli/nl/train_concatenated/train_facebook_voxpopuli_manifest_whisper_large_v3_norm_upprev_wer_filt_zipped_jz.json
#     $input_root/espnet/yodas/nl000/train_concatenated/train_espnet_yodas_manifest_whisper_large_v3_norm_upprev_wer_filt_zipped_jz.json
#     $input_root/espnet/yodas/nl100/train_concatenated/train_espnet_yodas_manifest_whisper_large_v3_norm_upprev_wer_filt_zipped_jz.json
    # # en
    # $input_root/mozilla-foundation/common_voice_17_0/en/train_concatenated/train_mozilla-foundation_common_voice_17_0_manifest_whisper_large_v3_norm_wer_filt_wer_zipped.json
    # $input_root/openslr/librispeech_asr/train_concatenated/train_openslr_librispeech_asr_manifest_whisper_large_v3_norm_upprev_wer_filt_zipped.json
    # $input_root/facebook/voxpopuli/en/train_concatenated/train_facebook_voxpopuli_manifest_whisper_large_v3_norm_upprev_wer_filt_zipped.json
    # $input_root/LIUM/tedlium/release3/train_concatenated/train_LIUM_tedlium_manifest_whisper_large_v3_norm_upprev_wer_filt_zipped.json
    # $input_root/MLCommons/peoples_speech/default/train_concatenated/train_MLCommons_peoples_speech_manifest_whisper_large_v3_norm_upprev_wer_filt_zipped.json
    # $input_root/speechcolab/gigaspeech/l/train_concatenated/train_speechcolab_gigaspeech_manifest_whisper_large_v3_norm_upprev_wer_filt_zipped.json
    # $input_root/distil-whisper/ami-ihm/ihm/train_concatenated/train_distil-whisper_ami-ihm_manifest_whisper_large_v3_norm_upprev_wer_filt_zipped.json
    # $input_root/distil-whisper/ami-sdm/sdm/train_concatenated/train_distil-whisper_ami-sdm_manifest_whisper_large_v3_norm_upprev_wer_filt_zipped.json
    # $input_root/espnet/yodas/en000/train_concatenated/train_espnet_yodas_manifest_whisper_large_v3_norm_upprev_wer_filt_zipped.json
    # $input_root/espnet/yodas/en001/train_concatenated/train_espnet_yodas_manifest_whisper_large_v3_norm_upprev_wer_filt_zipped.json
    # $input_root/espnet/yodas/en002/train_concatenated/train_espnet_yodas_manifest_whisper_large_v3_norm_upprev_wer_filt_zipped.json
    # $input_root/espnet/yodas/en003/train_concatenated/train_espnet_yodas_manifest_whisper_large_v3_norm_upprev_wer_filt_zipped.json
    # $input_root/espnet/yodas/en004/train_concatenated/train_espnet_yodas_manifest_whisper_large_v3_norm_upprev_wer_filt_zipped.json
    # $input_root/espnet/yodas/en005/train_concatenated/train_espnet_yodas_manifest_whisper_large_v3_norm_upprev_wer_filt_zipped.json
    # $input_root/espnet/yodas/en100/train_concatenated/train_espnet_yodas_manifest_whisper_large_v3_norm_upprev_wer_filt_zipped.json
    # $input_root/espnet/yodas/en101/train_concatenated/train_espnet_yodas_manifest_whisper_large_v3_norm_upprev_wer_filt_zipped.json
    # $input_root/espnet/yodas/en102/train_concatenated/train_espnet_yodas_manifest_whisper_large_v3_norm_upprev_wer_filt_zipped.json
    # $input_root/espnet/yodas/en103/train_concatenated/train_espnet_yodas_manifest_whisper_large_v3_norm_upprev_wer_filt_zipped.json
    # $input_root/espnet/yodas/en104/train_concatenated/train_espnet_yodas_manifest_whisper_large_v3_norm_upprev_wer_filt_zipped.json
    # $input_root/espnet/yodas/en105/train_concatenated/train_espnet_yodas_manifest_whisper_large_v3_norm_upprev_wer_filt_zipped.json
    # $input_root/espnet/yodas/en106/train_concatenated/train_espnet_yodas_manifest_whisper_large_v3_norm_upprev_wer_filt_zipped.json
    # $input_root/espnet/yodas/en107/train_concatenated/train_espnet_yodas_manifest_whisper_large_v3_norm_upprev_wer_filt_zipped.json
    # $input_root/espnet/yodas/en109/train_concatenated/train_espnet_yodas_manifest_whisper_large_v3_norm_upprev_wer_filt_zipped.json
    # $input_root/espnet/yodas/en110/train_concatenated/train_espnet_yodas_manifest_whisper_large_v3_norm_upprev_wer_filt_zipped.json
    # $input_root/espnet/yodas/en111/train_concatenated/train_espnet_yodas_manifest_whisper_large_v3_norm_upprev_wer_filt_zipped.json
    # $input_root/espnet/yodas/en112/train_concatenated/train_espnet_yodas_manifest_whisper_large_v3_norm_upprev_wer_filt_zipped.json
    # $input_root/espnet/yodas/en113/train_concatenated/train_espnet_yodas_manifest_whisper_large_v3_norm_upprev_wer_filt_zipped.json
    # $input_root/espnet/yodas/en114/train_concatenated/train_espnet_yodas_manifest_whisper_large_v3_norm_upprev_wer_filt_zipped.json
    # $input_root/espnet/yodas/en115/train_concatenated/train_espnet_yodas_manifest_whisper_large_v3_norm_upprev_wer_filt_zipped.json
    # $input_root/espnet/yodas/en116/train_concatenated/train_espnet_yodas_manifest_whisper_large_v3_norm_upprev_wer_filt_zipped.json
    # $input_root/espnet/yodas/en117/train_concatenated/train_espnet_yodas_manifest_whisper_large_v3_norm_upprev_wer_filt_zipped.json
    # $input_root/espnet/yodas/en118/train_concatenated/train_espnet_yodas_manifest_whisper_large_v3_norm_upprev_wer_filt_zipped.json
    # $input_root/espnet/yodas/en119/train_concatenated/train_espnet_yodas_manifest_whisper_large_v3_norm_upprev_wer_filt_zipped.json
    # $input_root/espnet/yodas/en120/train_concatenated/train_espnet_yodas_manifest_whisper_large_v3_norm_upprev_wer_filt_zipped.json
    # $input_root/espnet/yodas/en121/train_concatenated/train_espnet_yodas_manifest_whisper_large_v3_norm_upprev_wer_filt_zipped.json
    # $input_root/espnet/yodas/en122/train_concatenated/train_espnet_yodas_manifest_whisper_large_v3_norm_upprev_wer_filt_zipped.json
    # $input_root/espnet/yodas/en123/train_concatenated/train_espnet_yodas_manifest_whisper_large_v3_norm_upprev_wer_filt_zipped.json
    # $input_root/espnet/yodas/en124/train_concatenated/train_espnet_yodas_manifest_whisper_large_v3_norm_upprev_wer_filt_zipped.json
    # $input_root/espnet/yodas/en125/train_concatenated/train_espnet_yodas_manifest_whisper_large_v3_norm_upprev_wer_filt_zipped.json
    # $input_root/espnet/yodas/en126/train_concatenated/train_espnet_yodas_manifest_whisper_large_v3_norm_upprev_wer_filt_zipped.json
    # $input_root/espnet/yodas/en127/train_concatenated/train_espnet_yodas_manifest_whisper_large_v3_norm_upprev_wer_filt_zipped.json
    # $input_root/tmp/tmp_en_10300h.json
    # $input_root/tmp/tmp_non_en_wer20_5000h.json
# )
# input_files=$(IFS=+; echo "${input_files[*]}")

# downsample dataset
####################################################################################################
# output_file=$input_root/tmp/tmp_non_en_10000h.json
# # output_file=$input_root/tmp/tmp_en_7500h.json
# python scripts/downsample_data.py $input_files $output_file 10000

# check presence
####################################################################################################
# for input_file in "${input_files[@]}"; do
#     python scripts/iterate_audio_files.py $input_file audio_zip_filepath 320 || exit 1
# done

####################################################################################################
# input_file=/lustre/fsn1/projects/rech/gkb/commun/corpus/speech/cs/mozilla-foundation/common_voice_17_0/it/train/train_mozilla-foundation_common_voice_17_0_manifest_whisper_large_v3_norm_wer_filt.json
# lang=it
# python scripts/insert_column.py $input_file $input_file _language $lang

# input_file=/lustre/fsn1/projects/rech/gkb/commun/corpus/speech/cs/mozilla-foundation/common_voice_17_0/fr/train/train_mozilla-foundation_common_voice_17_0_manifest_whisper_large_v3_norm_wer_filt.json
# lang=fr
# python scripts/insert_column.py $input_file $input_file _language $lang

# input_file=/lustre/fsn1/projects/rech/gkb/commun/corpus/speech/cs/mozilla-foundation/common_voice_17_0/es/train/train_mozilla-foundation_common_voice_17_0_manifest_whisper_large_v3_norm_wer_filt.json
# lang=es
# python scripts/insert_column.py $input_file $input_file _language $lang

# input_file=/lustre/fsn1/projects/rech/gkb/commun/corpus/speech/cs/mozilla-foundation/common_voice_17_0/pt/train/train_mozilla-foundation_common_voice_17_0_manifest_whisper_large_v3_norm_wer_filt.json
# lang=pt
# python scripts/insert_column.py $input_file $input_file _language $lang

# input_file=/lustre/fsn1/projects/rech/gkb/commun/corpus/speech/cs/mozilla-foundation/common_voice_17_0/de/train/train_mozilla-foundation_common_voice_17_0_manifest_whisper_large_v3_norm_wer_filt.json
# lang=de
# python scripts/insert_column.py $input_file $input_file _language $lang

# input_file=/lustre/fsn1/projects/rech/gkb/commun/corpus/speech/cs/mozilla-foundation/common_voice_17_0/nl/train/train_mozilla-foundation_common_voice_17_0_manifest_whisper_large_v3_norm_wer_filt.json
# lang=nl
# python scripts/insert_column.py $input_file $input_file _language $lang

# input_file=/lustre/fsn1/projects/rech/gkb/commun/corpus/speech/cs/mozilla-foundation/common_voice_17_0/en/train/train_mozilla-foundation_common_voice_17_0_manifest_whisper_large_v3_norm_wer_filt.json
# lang=en
# python scripts/insert_column.py $input_file $input_file _language $lang

# create and postprocess code-switching data
####################################################################################################
# input_root=/lustre/fsn1/projects/rech/gkb/commun/corpus/speech/cs
# input_files=(
    # # ${input_root}/mozilla-foundation/common_voice_17_0/it/train/train_mozilla-foundation_common_voice_17_0_manifest_whisper_large_v3_norm_wer_filt.json
    # ${input_root}/mozilla-foundation/common_voice_17_0/fr/train/train_mozilla-foundation_common_voice_17_0_manifest_whisper_large_v3_norm_wer_filt.json
    # ${input_root}/mozilla-foundation/common_voice_17_0/es/train/train_mozilla-foundation_common_voice_17_0_manifest_whisper_large_v3_norm_wer_filt.json
    # # ${input_root}/mozilla-foundation/common_voice_17_0/pt/train/train_mozilla-foundation_common_voice_17_0_manifest_whisper_large_v3_norm_wer_filt.json
    # ${input_root}/mozilla-foundation/common_voice_17_0/de/train/train_mozilla-foundation_common_voice_17_0_manifest_whisper_large_v3_norm_wer_filt.json
    # # ${input_root}/mozilla-foundation/common_voice_17_0/nl/train/train_mozilla-foundation_common_voice_17_0_manifest_whisper_large_v3_norm_wer_filt.json
    # ${input_root}/mozilla-foundation/common_voice_17_0/en/train/train_mozilla-foundation_common_voice_17_0_manifest_whisper_large_v3_norm_wer_filt.json
    # yodas
    # ${input_root}/espnet/yodas/it000/train_concatenated/train_espnet_yodas_manifest_whisper_large_v3_norm_upprev_wer_filt.json
#     ${input_root}/espnet/yodas/fr000/train_concatenated/train_espnet_yodas_manifest_whisper_large_v3_norm_upprev_wer_filt.json
#     ${input_root}/espnet/yodas/es000/train_concatenated/train_espnet_yodas_manifest_whisper_large_v3_norm_upprev_wer_filt.json
#     # ${input_root}/espnet/yodas/pt000/train_concatenated/train_espnet_yodas_manifest_whisper_large_v3_norm_upprev_wer_filt.json
#     ${input_root}/espnet/yodas/de000/train_concatenated/train_espnet_yodas_manifest_whisper_large_v3_norm_upprev_wer_filt.json
#     # ${input_root}/espnet/yodas/nl000/train_concatenated/train_espnet_yodas_manifest_whisper_large_v3_norm_upprev_wer_filt.json
#     ${input_root}/espnet/yodas/en000/train_concatenated/train_espnet_yodas_manifest_whisper_large_v3_norm_upprev_wer_filt.json
# )
# input_files=$(IFS=+; echo "${input_files[*]}")

# output_root=/lustre/fsn1/projects/rech/gkb/commun/corpus/speech/stt-pseudo-labeled-whisper-large-v3-multilingual

# # output_audio_dir=${output_root}/code-switching/common_voice_17_0_4langs/train_concatenated
# # output_file=${output_audio_dir}/train_mozilla-foundation_common_voice_17_0_manifest_whisper_large_v3_norm_wer_filt.json
# # speaker_column_name=client_id
# output_audio_dir=${output_root}/code-switching/yodas_4langs/train_concatenated
# output_file=${output_audio_dir}/train_espnet_yodas_manifest_whisper_large_v3_norm_upprev_wer_filt.json
# speaker_column_name=speaker_id

# python scripts/create_code_switching_data.py \
#     --input_files $input_files \
#     --output_file $output_file \
#     --output_audio_dir $output_audio_dir \
#     --speaker_column_name $speaker_column_name \
#     --num_workers $num_workers \

# input_file=$output_file
# input_dir="${input_file%/*}"

# # wer
# python scripts/compute_wer.py \
#     --input_file_path "$input_file" \
#     --output_file_path "${input_file%.*}_wer.json" \
#     --num_workers $num_workers

# # python scripts/split_folder.py \
# #     --input_file_path "${input_file%.*}_wer.json" \
# #     --output_file_path "${input_file%.*}_wer.json" \
# #     --num_workers $num_workers

# # zipping audio files
# python scripts/zip_audio_files.py \
#     --input_file_path "${input_file%.*}_wer.json" \
#     --output_file_path "${input_file%.*}_wer_zipped.json" \
#     --num_workers $num_workers

# # tmp: remove unzipped audio files
# find $input_dir -mindepth 1 -type f -name "*.wav" -delete
# find $input_dir -mindepth 1 -type d -empty -delete

# python scripts/tmp.py \
#     --input_file "${input_file%.*}_wer_zipped.json" \
#     --output_file "${input_file%.*}_wer_zipped_tl.json" \
#     --num_workers $num_workers

####################################################################################################

# input_root=/lustre/fsn1/projects/rech/gkb/commun/corpus/speech/stt-pseudo-labeled-whisper-large-v3-multilingual
# input_file=${input_root}/MLCommons/peoples_speech/clean_sa/train/train_MLCommons_peoples_speech_manifest.json
# input_file=${input_root}/MLCommons/peoples_speech/dirty_sa/train/train_MLCommons_peoples_speech_manifest.json
# input_file=${input_root}/MLCommons/peoples_speech/dirty/train/train_MLCommons_peoples_speech_manifest.json
# python scripts/split_folder.py \
#     --input_file_path "$input_file" \
#     --output_file_path "$input_file" \
#     --num_workers $num_workers

####################################################################################################
# input_root=/lustre/fsn1/projects/rech/gkb/commun/corpus/speech/stt-pseudo-labeled-whisper-large-v3-multilingual
# input_files=(
    # # it
    # $input_root/mozilla-foundation/common_voice_17_0/it/train_concatenated/train_mozilla-foundation_common_voice_17_0_manifest_whisper_large_v3_norm_wer_filt_wer_zipped_jz.json
    # $input_root/facebook/multilingual_librispeech/italian/train_concatenated/train_facebook_multilingual_librispeech_manifest_whisper_large_v3_norm_upprev_wer_filt_zipped_jz.json
    # $input_root/facebook/voxpopuli/it/train_concatenated/train_facebook_voxpopuli_manifest_whisper_large_v3_norm_upprev_wer_filt_zipped_jz.json
    # $input_root/multilingual-tedx/it-it/train_concatenated/train_mtedx_asr_manifest_whisper_large_v3_norm_upprev_wer_filt_zipped_jz.json
    # $input_root/espnet/yodas/it000/train_concatenated/train_espnet_yodas_manifest_whisper_large_v3_norm_upprev_wer_filt_zipped_jz.json
    # $input_root/espnet/yodas/it100/train_concatenated/train_espnet_yodas_manifest_whisper_large_v3_norm_upprev_wer_filt_zipped_jz.json
    # $input_root/espnet/yodas/it101/train_concatenated/train_espnet_yodas_manifest_whisper_large_v3_norm_upprev_wer_filt_zipped_jz.json
    # # fr
    # $input_root/mozilla-foundation/common_voice_17_0/fr/train_concatenated/train_mozilla-foundation_common_voice_17_0_manifest_whisper_large_v3_norm_wer_filt_wer_zipped_jz.json
    # $input_root/facebook/multilingual_librispeech/french/train_concatenated/train_facebook_multilingual_librispeech_manifest_whisper_large_v3_norm_upprev_wer_filt_zipped_jz.json
    # $input_root/facebook/voxpopuli/fr/train_concatenated/train_facebook_voxpopuli_manifest_whisper_large_v3_norm_upprev_wer_filt_zipped_jz.json
    # $input_root/multilingual-tedx/fr-fr/train_concatenated/train_mtedx_asr_manifest_whisper_large_v3_norm_upprev_wer_filt_zipped_jz.json
    # $input_root/espnet/yodas/fr000/train_concatenated/train_espnet_yodas_manifest_whisper_large_v3_norm_upprev_wer_filt_zipped_jz.json
    # $input_root/espnet/yodas/fr100/train_concatenated/train_espnet_yodas_manifest_whisper_large_v3_norm_upprev_wer_filt_zipped_jz.json
    # $input_root/espnet/yodas/fr101/train_concatenated/train_espnet_yodas_manifest_whisper_large_v3_norm_upprev_wer_filt_zipped_jz.json
    # $input_root/espnet/yodas/fr102/train_concatenated/train_espnet_yodas_manifest_whisper_large_v3_norm_upprev_wer_filt_zipped_jz.json
    # $input_root/espnet/yodas/fr103/train_concatenated/train_espnet_yodas_manifest_whisper_large_v3_norm_upprev_wer_filt_zipped_jz.json
    # # es
    # $input_root/mozilla-foundation/common_voice_17_0/es/train_concatenated/train_mozilla-foundation_common_voice_17_0_manifest_whisper_large_v3_norm_wer_filt_wer_zipped_jz.json
    # $input_root/facebook/multilingual_librispeech/spanish/train_concatenated/train_facebook_multilingual_librispeech_manifest_whisper_large_v3_norm_upprev_wer_filt_zipped_jz.json
    # $input_root/facebook/voxpopuli/es/train_concatenated/train_facebook_voxpopuli_manifest_whisper_large_v3_norm_upprev_wer_filt_zipped_jz.json
    # $input_root/multilingual-tedx/es-es/train_concatenated/train_mtedx_asr_manifest_whisper_large_v3_norm_upprev_wer_filt_zipped_jz.json
    # $input_root/espnet/yodas/es000/train_concatenated/train_espnet_yodas_manifest_whisper_large_v3_norm_upprev_wer_filt_zipped_jz.json
    # $input_root/espnet/yodas/es100/train_concatenated/train_espnet_yodas_manifest_whisper_large_v3_norm_upprev_wer_filt_zipped_jz.json
    # $input_root/espnet/yodas/es101/train_concatenated/train_espnet_yodas_manifest_whisper_large_v3_norm_upprev_wer_filt_zipped_jz.json
    # # pt
    # $input_root/mozilla-foundation/common_voice_17_0/pt/train_concatenated/train_mozilla-foundation_common_voice_17_0_manifest_whisper_large_v3_norm_wer_filt_wer_zipped_jz.json
    # $input_root/facebook/multilingual_librispeech/portuguese/train_concatenated/train_facebook_multilingual_librispeech_manifest_whisper_large_v3_norm_upprev_wer_filt_zipped_jz.json
    # $input_root/multilingual-tedx/pt-pt/train_concatenated/train_mtedx_asr_manifest_whisper_large_v3_norm_upprev_wer_filt_zipped_jz.json
    # $input_root/espnet/yodas/pt000/train_concatenated/train_espnet_yodas_manifest_whisper_large_v3_norm_upprev_wer_filt_zipped_jz.json
    # $input_root/espnet/yodas/pt100/train_concatenated/train_espnet_yodas_manifest_whisper_large_v3_norm_upprev_wer_filt_zipped_jz.json
    # $input_root/espnet/yodas/pt101/train_concatenated/train_espnet_yodas_manifest_whisper_large_v3_norm_upprev_wer_filt_zipped_jz.json
    # $input_root/espnet/yodas/pt102/train_concatenated/train_espnet_yodas_manifest_whisper_large_v3_norm_upprev_wer_filt_zipped_jz.json
    # $input_root/espnet/yodas/pt103/train_concatenated/train_espnet_yodas_manifest_whisper_large_v3_norm_upprev_wer_filt_zipped_jz.json
    # # de
    # $input_root/mozilla-foundation/common_voice_17_0/de/train_concatenated/train_mozilla-foundation_common_voice_17_0_manifest_whisper_large_v3_norm_wer_filt_wer_zipped_jz.json
    # $input_root/facebook/multilingual_librispeech/german/train_concatenated/train_facebook_multilingual_librispeech_manifest_whisper_large_v3_norm_upprev_wer_filt_zipped_jz.json
    # $input_root/facebook/voxpopuli/de/train_concatenated/train_facebook_voxpopuli_manifest_whisper_large_v3_norm_upprev_wer_filt_zipped_jz.json
    # $input_root/multilingual-tedx/de-de/train_concatenated/train_mtedx_asr_manifest_whisper_large_v3_norm_upprev_wer_filt_zipped_jz.json
    # $input_root/espnet/yodas/de000/train_concatenated/train_espnet_yodas_manifest_whisper_large_v3_norm_upprev_wer_filt_zipped_jz.json
    # $input_root/espnet/yodas/de100/train_concatenated/train_espnet_yodas_manifest_whisper_large_v3_norm_upprev_wer_filt_zipped_jz.json
    # $input_root/espnet/yodas/de101/train_concatenated/train_espnet_yodas_manifest_whisper_large_v3_norm_upprev_wer_filt_zipped_jz.json
    # $input_root/espnet/yodas/de102/train_concatenated/train_espnet_yodas_manifest_whisper_large_v3_norm_upprev_wer_filt_zipped_jz.json
    # # nl
    # $input_root/mozilla-foundation/common_voice_17_0/nl/train_concatenated/train_mozilla-foundation_common_voice_17_0_manifest_whisper_large_v3_norm_wer_filt_wer_zipped_jz.json
    # $input_root/facebook/multilingual_librispeech/dutch/train_concatenated/train_facebook_multilingual_librispeech_manifest_whisper_large_v3_norm_upprev_wer_filt_zipped_jz.json
    # $input_root/facebook/voxpopuli/nl/train_concatenated/train_facebook_voxpopuli_manifest_whisper_large_v3_norm_upprev_wer_filt_zipped_jz.json
    # $input_root/espnet/yodas/nl000/train_concatenated/train_espnet_yodas_manifest_whisper_large_v3_norm_upprev_wer_filt_zipped_jz.json
    # $input_root/espnet/yodas/nl100/train_concatenated/train_espnet_yodas_manifest_whisper_large_v3_norm_upprev_wer_filt_zipped_jz.json
    # # en
    # $input_root/mozilla-foundation/common_voice_17_0/en/train_concatenated/train_mozilla-foundation_common_voice_17_0_manifest_whisper_large_v3_norm_wer_filt_wer_zipped.json
    # $input_root/openslr/librispeech_asr/train_concatenated/train_openslr_librispeech_asr_manifest_whisper_large_v3_norm_upprev_wer_filt_zipped.json
    # $input_root/facebook/voxpopuli/en/train_concatenated/train_facebook_voxpopuli_manifest_whisper_large_v3_norm_upprev_wer_filt_zipped.json
    # $input_root/LIUM/tedlium/release3/train_concatenated/train_LIUM_tedlium_manifest_whisper_large_v3_norm_upprev_wer_filt_zipped.json
    # $input_root/MLCommons/peoples_speech/default/train_concatenated/train_MLCommons_peoples_speech_manifest_whisper_large_v3_norm_upprev_wer_filt_zipped.json
    # $input_root/MLCommons/peoples_speech/clean_sa/train_concatenated/train_MLCommons_peoples_speech_manifest_whisper_large_v3_norm_upprev_wer_filt_zipped.json
    # $input_root/MLCommons/peoples_speech/dirty/train_concatenated/train_MLCommons_peoples_speech_manifest_whisper_large_v3_norm_upprev_wer_filt_zipped.json
    # $input_root/MLCommons/peoples_speech/dirty_sa/train_concatenated/train_MLCommons_peoples_speech_manifest_whisper_large_v3_norm_upprev_wer_filt_zipped.json
    # $input_root/speechcolab/gigaspeech/l/train_concatenated/train_speechcolab_gigaspeech_manifest_whisper_large_v3_norm_upprev_wer_filt_zipped.json
    # $input_root/distil-whisper/ami-ihm/ihm/train_concatenated/train_distil-whisper_ami-ihm_manifest_whisper_large_v3_norm_upprev_wer_filt_zipped.json
    # $input_root/distil-whisper/ami-sdm/sdm/train_concatenated/train_distil-whisper_ami-sdm_manifest_whisper_large_v3_norm_upprev_wer_filt_zipped.json
    # $input_root/espnet/yodas/en000/train_concatenated/train_espnet_yodas_manifest_whisper_large_v3_norm_upprev_wer_filt_zipped.json
    # $input_root/espnet/yodas/en001/train_concatenated/train_espnet_yodas_manifest_whisper_large_v3_norm_upprev_wer_filt_zipped.json
    # $input_root/espnet/yodas/en002/train_concatenated/train_espnet_yodas_manifest_whisper_large_v3_norm_upprev_wer_filt_zipped.json
    # $input_root/espnet/yodas/en003/train_concatenated/train_espnet_yodas_manifest_whisper_large_v3_norm_upprev_wer_filt_zipped.json
    # $input_root/espnet/yodas/en004/train_concatenated/train_espnet_yodas_manifest_whisper_large_v3_norm_upprev_wer_filt_zipped.json
    # $input_root/espnet/yodas/en005/train_concatenated/train_espnet_yodas_manifest_whisper_large_v3_norm_upprev_wer_filt_zipped.json
    # $input_root/espnet/yodas/en100/train_concatenated/train_espnet_yodas_manifest_whisper_large_v3_norm_upprev_wer_filt_zipped.json
    # $input_root/espnet/yodas/en101/train_concatenated/train_espnet_yodas_manifest_whisper_large_v3_norm_upprev_wer_filt_zipped.json
    # $input_root/espnet/yodas/en102/train_concatenated/train_espnet_yodas_manifest_whisper_large_v3_norm_upprev_wer_filt_zipped.json
    # $input_root/espnet/yodas/en103/train_concatenated/train_espnet_yodas_manifest_whisper_large_v3_norm_upprev_wer_filt_zipped.json
    # $input_root/espnet/yodas/en104/train_concatenated/train_espnet_yodas_manifest_whisper_large_v3_norm_upprev_wer_filt_zipped.json
    # $input_root/espnet/yodas/en105/train_concatenated/train_espnet_yodas_manifest_whisper_large_v3_norm_upprev_wer_filt_zipped.json
    # $input_root/espnet/yodas/en106/train_concatenated/train_espnet_yodas_manifest_whisper_large_v3_norm_upprev_wer_filt_zipped.json
    # $input_root/espnet/yodas/en107/train_concatenated/train_espnet_yodas_manifest_whisper_large_v3_norm_upprev_wer_filt_zipped.json
    # $input_root/espnet/yodas/en109/train_concatenated/train_espnet_yodas_manifest_whisper_large_v3_norm_upprev_wer_filt_zipped.json
    # $input_root/espnet/yodas/en110/train_concatenated/train_espnet_yodas_manifest_whisper_large_v3_norm_upprev_wer_filt_zipped.json
    # $input_root/espnet/yodas/en111/train_concatenated/train_espnet_yodas_manifest_whisper_large_v3_norm_upprev_wer_filt_zipped.json
    # $input_root/espnet/yodas/en112/train_concatenated/train_espnet_yodas_manifest_whisper_large_v3_norm_upprev_wer_filt_zipped.json
    # $input_root/espnet/yodas/en113/train_concatenated/train_espnet_yodas_manifest_whisper_large_v3_norm_upprev_wer_filt_zipped.json
    # $input_root/espnet/yodas/en114/train_concatenated/train_espnet_yodas_manifest_whisper_large_v3_norm_upprev_wer_filt_zipped.json
    # $input_root/espnet/yodas/en115/train_concatenated/train_espnet_yodas_manifest_whisper_large_v3_norm_upprev_wer_filt_zipped.json
    # $input_root/espnet/yodas/en116/train_concatenated/train_espnet_yodas_manifest_whisper_large_v3_norm_upprev_wer_filt_zipped.json
    # $input_root/espnet/yodas/en117/train_concatenated/train_espnet_yodas_manifest_whisper_large_v3_norm_upprev_wer_filt_zipped.json
    # $input_root/espnet/yodas/en118/train_concatenated/train_espnet_yodas_manifest_whisper_large_v3_norm_upprev_wer_filt_zipped.json
    # $input_root/espnet/yodas/en119/train_concatenated/train_espnet_yodas_manifest_whisper_large_v3_norm_upprev_wer_filt_zipped.json
    # $input_root/espnet/yodas/en120/train_concatenated/train_espnet_yodas_manifest_whisper_large_v3_norm_upprev_wer_filt_zipped.json
    # $input_root/espnet/yodas/en121/train_concatenated/train_espnet_yodas_manifest_whisper_large_v3_norm_upprev_wer_filt_zipped.json
    # $input_root/espnet/yodas/en122/train_concatenated/train_espnet_yodas_manifest_whisper_large_v3_norm_upprev_wer_filt_zipped.json
    # $input_root/espnet/yodas/en123/train_concatenated/train_espnet_yodas_manifest_whisper_large_v3_norm_upprev_wer_filt_zipped.json
    # $input_root/espnet/yodas/en124/train_concatenated/train_espnet_yodas_manifest_whisper_large_v3_norm_upprev_wer_filt_zipped.json
    # $input_root/espnet/yodas/en125/train_concatenated/train_espnet_yodas_manifest_whisper_large_v3_norm_upprev_wer_filt_zipped.json
    # $input_root/espnet/yodas/en126/train_concatenated/train_espnet_yodas_manifest_whisper_large_v3_norm_upprev_wer_filt_zipped.json
    # $input_root/espnet/yodas/en127/train_concatenated/train_espnet_yodas_manifest_whisper_large_v3_norm_upprev_wer_filt_zipped.json
    ## cs
    # $input_root/code-switching/common_voice_17_0/train_mozilla-foundation_common_voice_17_0_manifest_whisper_large_v3_norm_wer_filt_wer_zipped_tl.json
    # $input_root/code-switching/yodas/train_espnet_yodas_manifest_whisper_large_v3_norm_upprev_wer_filt_wer_zipped_tl.json
    # $input_root/code-switching/yodas/train_concatenated/train_espnet_yodas_manifest_whisper_large_v3_norm_upprev_wer_filt_wer_zipped_tl.json
    ## translation
    # $input_root/espnet/yodas/de000/train_concatenated/train_espnet_yodas_manifest_whisper_large_v3_norm_upprev_wer_filt_zipped_jz_translated_processed.json
    # $input_root/espnet/yodas/pt000/train_concatenated/train_espnet_yodas_manifest_whisper_large_v3_norm_upprev_wer_filt_zipped_jz_translated_processed.json
    # $input_root/espnet/yodas/es000/train_concatenated/train_espnet_yodas_manifest_whisper_large_v3_norm_upprev_wer_filt_zipped_jz_translated_processed.json
    # $input_root/espnet/yodas/fr000/train_concatenated/train_espnet_yodas_manifest_whisper_large_v3_norm_upprev_wer_filt_zipped_jz_translated_processed.json
    # $input_root/espnet/yodas/it000/train_concatenated/train_espnet_yodas_manifest_whisper_large_v3_norm_upprev_wer_filt_zipped_jz_translated_processed.json
    # $input_root/espnet/yodas/nl000/train_concatenated/train_espnet_yodas_manifest_whisper_large_v3_norm_upprev_wer_filt_zipped_jz_translated_processed.json
    ## merged
    # $input_root/final/train_manifest_multi_wer20.json
    # $input_root/final/train_manifest_fr_wer20.json
    # $input_root/final/train_manifest_es_wer20.json
    # $input_root/final/train_manifest_de_wer20.json
    # $input_root/final/train_manifest_it_wer20.json
    # $input_root/final/train_manifest_pt_wer20.json
    # $input_root/final/train_manifest_nl_wer20.json
    # $input_root/final/train_manifest_en_wer10.json
    # $input_root/final/train_manifest_en_wer10_subset_46kh.json
    # $input_root/final/train_manifest_en_wer10_subset_23kh.json
    # $input_root/final/train_manifest_cs_wer10.json
    # valid
    # $input_root/mozilla-foundation/common_voice_17_0/it/validation/validation_mozilla-foundation_common_voice_17_0_manifest.json
    # $input_root/mozilla-foundation/common_voice_17_0/fr/validation/validation_mozilla-foundation_common_voice_17_0_manifest.json
    # $input_root/mozilla-foundation/common_voice_17_0/es/validation/validation_mozilla-foundation_common_voice_17_0_manifest.json
    # $input_root/mozilla-foundation/common_voice_17_0/pt/validation/validation_mozilla-foundation_common_voice_17_0_manifest.json
    # $input_root/mozilla-foundation/common_voice_17_0/de/validation/validation_mozilla-foundation_common_voice_17_0_manifest.json
    # $input_root/mozilla-foundation/common_voice_17_0/nl/validation/validation_mozilla-foundation_common_voice_17_0_manifest.json
    # $input_root/mozilla-foundation/common_voice_17_0/en/validation/validation_mozilla-foundation_common_voice_17_0_manifest.json
    # $input_root/google/fleurs/it_it/validation/validation_google_fleurs_manifest.json
    # $input_root/google/fleurs/fr_fr/validation/validation_google_fleurs_manifest.json
    # $input_root/google/fleurs/es_419/validation/validation_google_fleurs_manifest.json
    # $input_root/google/fleurs/pt_br/validation/validation_google_fleurs_manifest.json
    # $input_root/google/fleurs/de_de/validation/validation_google_fleurs_manifest.json
    # $input_root/google/fleurs/nl_nl/validation/validation_google_fleurs_manifest.json
    # $input_root/google/fleurs/en_us/validation/validation_google_fleurs_manifest.json
# )

# update suffix
# input_files=(
#     "${input_files[@]/.json/_pnc_l31_processed.json}"
# )

# input_files=$(IFS=+; echo "${input_files[*]}")

# merge train files
####################################################################################################

# output_file=$input_root/final/train_manifest_en_wer10.json
# output_file=$input_root/final/train_manifest_en_wer10_subset_46kh.json
# output_file=$input_root/final/train_manifest_en_wer10_subset_30kh.json
# output_file=$input_root/final/train_manifest_en_wer10_subset_23kh.json
# output_file=$input_root/final/train_manifest_multi_wer20.json
# output_file=$input_root/final/train_manifest_nl_wer20.json
# output_file=$input_root/final/train_manifest_cs_wer10.json
# output_file=$input_root/final_new/train_manifest_translation_wer10.json

# python scripts/downsample_data.py $input_files $output_file 30000

# output_file=$input_root/final_nemo/train_manifest_it_wer20.json
# output_file=$input_root/final_nemo/train_manifest_fr_wer20.json
# output_file=$input_root/final_nemo/train_manifest_es_wer20.json
# output_file=$input_root/final_nemo/train_manifest_pt_wer20.json
# output_file=$input_root/final_nemo/train_manifest_de_wer20.json
# output_file=$input_root/final_nemo/train_manifest_nl_wer20.json
# output_file=$input_root/final_nemo/train_manifest_en_wer10.json

# python scripts/downsample_data.py $input_files $output_file 30000


####################################################################################################

input_root=/lustre/fsn1/projects/rech/gkb/commun/corpus/speech/stt-pseudo-labeled-whisper-large-v3-multilingual

input_files=(
    $input_root/mozilla-foundation/common_voice_17_0/it/validation/validation_mozilla-foundation_common_voice_17_0_manifest_nm.json
    $input_root/mozilla-foundation/common_voice_17_0/fr/validation/validation_mozilla-foundation_common_voice_17_0_manifest_nm.json
    $input_root/mozilla-foundation/common_voice_17_0/es/validation/validation_mozilla-foundation_common_voice_17_0_manifest_nm.json
    $input_root/mozilla-foundation/common_voice_17_0/pt/validation/validation_mozilla-foundation_common_voice_17_0_manifest_nm.json
    $input_root/mozilla-foundation/common_voice_17_0/de/validation/validation_mozilla-foundation_common_voice_17_0_manifest_nm.json
    $input_root/mozilla-foundation/common_voice_17_0/nl/validation/validation_mozilla-foundation_common_voice_17_0_manifest_nm.json
    $input_root/mozilla-foundation/common_voice_17_0/en/validation/validation_mozilla-foundation_common_voice_17_0_manifest_nm.json
    $input_root/google/fleurs/it_it/validation/validation_google_fleurs_manifest_nm.json
    $input_root/google/fleurs/fr_fr/validation/validation_google_fleurs_manifest_nm.json
    $input_root/google/fleurs/es_419/validation/validation_google_fleurs_manifest_nm.json
    $input_root/google/fleurs/pt_br/validation/validation_google_fleurs_manifest_nm.json
    $input_root/google/fleurs/de_de/validation/validation_google_fleurs_manifest_nm.json
    $input_root/google/fleurs/nl_nl/validation/validation_google_fleurs_manifest_nm.json
    $input_root/google/fleurs/en_us/validation/validation_google_fleurs_manifest_nm.json
)

for input_file in "${input_files[@]}"; do
    # add pnc, num2words
    ####################################################################################################
    # sbatch scripts/format_text.slurm $input_file
    # python scripts/postproc_formatting.py \
    #     --input_file_path ${input_file%.*}_pnc_l31.json \
    #     --output_file_path ${input_file%.*}_pnc_l31_processed.json \
    #     --num_workers 32

    # translation
    ####################################################################################################
    # sbatch scripts/translate_text.slurm $input_file
    # todo: filter by len ratio, update prev, add lang/task, check timestamps
    # python scripts/postproc_translation.py \
    #     --input_file_path ${input_file%.*}_translated.json \
    #     --output_file_path ${input_file%.*}_translated_processed.json \
    #     --num_workers 32

    # output_file="${input_file/\/final\//\/final_new\/}"
    # python scripts/insert_column.py $input_file $output_file _task transcribe 32

#     echo $input_file
#     exit

    # python scripts/insert_column.py $input_file ${input_file%.*}_nm.json lang en 32

    
    l=${input_file%/*}
    l=${l%/*}
    l=${l##*/}
    # echo $l
    # exit
    cp $input_file ${input_file%.*}_${l}.json

done

echo "END TIME: $(date)"
