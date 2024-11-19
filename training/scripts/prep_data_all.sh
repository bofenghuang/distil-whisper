#!/usr/bin/env bash

# export CUDA_VISIBLE_DEVICES="4,5,6,7"

n=8

gpuindex=0

# lang=fr
# lang=de
# lang=es
# lang=it
lang=pt
# lang=nl

# input_file=/projects/bhuang/corpus/speech/nemo_manifests/mozilla-foundation/common_voice_17_0/fr/train/train_mozilla-foundation_common_voice_17_0_manifest.json
# input_file=/projects/bhuang/corpus/speech/nemo_manifests/mozilla-foundation/common_voice_17_0/de/train/train_mozilla-foundation_common_voice_17_0_manifest.json
# input_file=/projects/bhuang/corpus/speech/nemo_manifests/mozilla-foundation/common_voice_17_0/es/train/train_mozilla-foundation_common_voice_17_0_manifest.json
# input_file=/projects/bhuang/corpus/speech/nemo_manifests/mozilla-foundation/common_voice_17_0/it/train/train_mozilla-foundation_common_voice_17_0_manifest.json
# input_file=/projects/bhuang/corpus/speech/nemo_manifests/mozilla-foundation/common_voice_17_0/nl/train/train_mozilla-foundation_common_voice_17_0_manifest.json
# input_file=/projects/bhuang/corpus/speech/nemo_manifests/mozilla-foundation/common_voice_17_0/pt/train/train_mozilla-foundation_common_voice_17_0_manifest.json

# ./scripts/run_prep_mcv.sh $input_file $lang $n $gpuindex

# fr
# input_files=(
#     /projects/bhuang/corpus/speech/nemo_manifests/facebook/multilingual_librispeech/french/train/train_facebook_multilingual_librispeech_manifest.json
#     /projects/bhuang/corpus/speech/nemo_manifests/facebook/voxpopuli/fr/train/train_facebook_voxpopuli_manifest.json
#     /projects/bhuang/corpus/speech/nemo_manifests/multilingual-tedx/fr-fr/train/train_mtedx_asr_manifest.json
#     /projects/bhuang/corpus/speech/nemo_manifests/espnet/yodas/fr000/train/train_espnet_yodas_manifest.json
#     /projects/bhuang/corpus/speech/nemo_manifests/espnet/yodas/fr100/train/train_espnet_yodas_manifest.json
#     /projects/bhuang/corpus/speech/nemo_manifests/espnet/yodas/fr101/train/train_espnet_yodas_manifest.json
#     /projects/bhuang/corpus/speech/nemo_manifests/espnet/yodas/fr102/train/train_espnet_yodas_manifest.json
#     /projects/bhuang/corpus/speech/nemo_manifests/espnet/yodas/fr103/train/train_espnet_yodas_manifest.json
# )

# de
# input_files=(
    # /projects/bhuang/corpus/speech/nemo_manifests/facebook/multilingual_librispeech/german/train/train_facebook_multilingual_librispeech_manifest.json
    # /projects/bhuang/corpus/speech/nemo_manifests/facebook/voxpopuli/de/train/train_facebook_voxpopuli_manifest.json
    # /projects/bhuang/corpus/speech/nemo_manifests/multilingual-tedx/de-de/train/train_mtedx_asr_manifest.json
    # /projects/bhuang/corpus/speech/nemo_manifests/espnet/yodas/de000/train/train_espnet_yodas_manifest.json
    # /projects/bhuang/corpus/speech/nemo_manifests/espnet/yodas/de100/train/train_espnet_yodas_manifest.json
    # /projects/bhuang/corpus/speech/nemo_manifests/espnet/yodas/de101/train/train_espnet_yodas_manifest.json
    # /projects/bhuang/corpus/speech/nemo_manifests/espnet/yodas/de102/train/train_espnet_yodas_manifest.json
# )

# es
# input_files=(
    # /projects/bhuang/corpus/speech/nemo_manifests/facebook/multilingual_librispeech/spanish/train/train_facebook_multilingual_librispeech_manifest.json
    # /projects/bhuang/corpus/speech/nemo_manifests/facebook/voxpopuli/es/train/train_facebook_voxpopuli_manifest.json
    # /projects/bhuang/corpus/speech/nemo_manifests/multilingual-tedx/es-es/train/train_mtedx_asr_manifest.json
    # /projects/bhuang/corpus/speech/nemo_manifests/espnet/yodas/es000/train/train_espnet_yodas_manifest.json
    # /projects/bhuang/corpus/speech/nemo_manifests/espnet/yodas/es100/train/train_espnet_yodas_manifest.json
    # /projects/bhuang/corpus/speech/nemo_manifests/espnet/yodas/es101/train/train_espnet_yodas_manifest.json
# )

# it
# input_files=(
#     /projects/bhuang/corpus/speech/nemo_manifests/facebook/multilingual_librispeech/italian/train/train_facebook_multilingual_librispeech_manifest.json
#     /projects/bhuang/corpus/speech/nemo_manifests/facebook/voxpopuli/it/train/train_facebook_voxpopuli_manifest.json
#     /projects/bhuang/corpus/speech/nemo_manifests/multilingual-tedx/it-it/train/train_mtedx_asr_manifest.json
#     /projects/bhuang/corpus/speech/nemo_manifests/espnet/yodas/it000/train/train_espnet_yodas_manifest.json
#     /projects/bhuang/corpus/speech/nemo_manifests/espnet/yodas/it100/train/train_espnet_yodas_manifest.json
#     /projects/bhuang/corpus/speech/nemo_manifests/espnet/yodas/it101/train/train_espnet_yodas_manifest.json
# )

# pt
input_files=(
    # /projects/bhuang/corpus/speech/nemo_manifests/facebook/multilingual_librispeech/portuguese/train/train_facebook_multilingual_librispeech_manifest.json
    # /projects/bhuang/corpus/speech/nemo_manifests/multilingual-tedx/pt-pt/train/train_mtedx_asr_manifest.json
    # /projects/bhuang/corpus/speech/nemo_manifests/espnet/yodas/pt000/train/train_espnet_yodas_manifest.json
    # /projects/bhuang/corpus/speech/nemo_manifests/espnet/yodas/pt100/train/train_espnet_yodas_manifest.json
    /projects/bhuang/corpus/speech/nemo_manifests/espnet/yodas/pt101/train/train_espnet_yodas_manifest.json
    # /projects/bhuang/corpus/speech/nemo_manifests/espnet/yodas/pt102/train/train_espnet_yodas_manifest.json
    /projects/bhuang/corpus/speech/nemo_manifests/espnet/yodas/pt103/train/train_espnet_yodas_manifest.json
)

# nl
# input_files=(
    # /projects/bhuang/corpus/speech/nemo_manifests/facebook/multilingual_librispeech/dutch/train/train_facebook_multilingual_librispeech_manifest.json
    # /projects/bhuang/corpus/speech/nemo_manifests/facebook/voxpopuli/nl/train/train_facebook_voxpopuli_manifest.json
    # /projects/bhuang/corpus/speech/nemo_manifests/espnet/yodas/nl000/train/train_espnet_yodas_manifest.json
    # /projects/bhuang/corpus/speech/nemo_manifests/espnet/yodas/nl100/train/train_espnet_yodas_manifest.json
# )

for input_file in "${input_files[@]}"; do
    ./scripts/run_prep.sh $input_file $lang $n $gpuindex
done


##############################################################################################################

# input_files=(
#     /projects/bhuang/corpus/speech/nemo_manifests/mozilla-foundation/common_voice_17_0/fr/train_concatenated/train_mozilla-foundation_common_voice_17_0_manifest_whisper_large_v3_norm_wer_filt_wer.json
#     /projects/bhuang/corpus/speech/nemo_manifests/facebook/multilingual_librispeech/french/train_concatenated/train_facebook_multilingual_librispeech_manifest_whisper_large_v3_norm_upprev_wer_filt.json
#     /projects/bhuang/corpus/speech/nemo_manifests/facebook/voxpopuli/fr/train_concatenated/train_facebook_voxpopuli_manifest_whisper_large_v3_norm_upprev_wer_filt.json
#     /projects/bhuang/corpus/speech/nemo_manifests/multilingual-tedx/fr-fr/train_concatenated/train_mtedx_asr_manifest_whisper_large_v3_norm_upprev_wer_filt.json
#     /projects/bhuang/corpus/speech/nemo_manifests/espnet/yodas/fr000/train_concatenated/train_espnet_yodas_manifest_whisper_large_v3_norm_upprev_wer_filt.json
#     /projects/bhuang/corpus/speech/nemo_manifests/espnet/yodas/fr100/train_concatenated/train_espnet_yodas_manifest_whisper_large_v3_norm_upprev_wer_filt.json
#     /projects/bhuang/corpus/speech/nemo_manifests/espnet/yodas/fr101/train_concatenated/train_espnet_yodas_manifest_whisper_large_v3_norm_upprev_wer_filt.json
#     /projects/bhuang/corpus/speech/nemo_manifests/espnet/yodas/fr102/train_concatenated/train_espnet_yodas_manifest_whisper_large_v3_norm_upprev_wer_filt.json
#     /projects/bhuang/corpus/speech/nemo_manifests/espnet/yodas/fr103/train_concatenated/train_espnet_yodas_manifest_whisper_large_v3_norm_upprev_wer_filt.json
# )

# for input_file in "${input_files[@]}"; do
#     # python scripts/get_duration_stats.py $input_file
#     python scripts/validate_audio_files_presence.py $input_file
# done

##############################################################################################################
