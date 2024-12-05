#!/usr/bin/env bash
# Copyright 2024  Bofeng Huang

# Run evaluation

set -x -e

echo "START TIME: $(date)"

# https://github.com/pytorch/audio/issues/1021#issuecomment-726915239
# export OMP_NUM_THREADS="1"

# py
myscriptspath="/home/bhuang/myscripts"
export PYTHONPATH="${PYTHONPATH:-}:$myscriptspath"
export PYTHONUNBUFFERED=1

# hf
# export HF_HOME="/projects/bhuang/.cache/huggingface"
# export TOKENIZERS_PARALLELISM="false"
# export BITSANDBYTES_NOWELCOME="1"
# export HF_HUB_ENABLE_HF_TRANSFER="1"
# export HF_HUB_OFFLINE="1"
# export HF_DATASETS_OFFLINE="1"
# export HF_EVALUATE_OFFLINE="1"

# cuda
# export CUDA_VISIBLE_DEVICES="4"

# load models
# multilingual
# model_name_or_path="openai/whisper-small"
# model_name_or_path="openai/whisper-large-v2"
# model_name_or_path="openai/whisper-large-v3"

# en
# model_name_or_path="distil-whisper/distil-large-v3"

# it
# model_name_or_path="/projects/bhuang/models/asr/whisper/it/whisper-large-v3-distil-it"

# fr
# model_name_or_path="eustlb/distil-large-v3-fr"
# model_name_or_path="/projects/bhuang/models/asr/whisper/fr/whisper-large-v3-distil-fr"

# take args
model_name_or_path=$1
lang=$2

# assistant_model_name_or_path="bofenghuang/whisper-large-v3-french-distil-dec2"

# language
# lang="$lang"
# lang="french"
# lang="italian"

output_root_dir="./outputs/evaluations/multilingual/$lang"
# output_root_dir=$2

tmp_model_id="$(echo "${model_name_or_path##*/}" | sed -e "s/[ |=/-]/_/g")"
outdir="$output_root_dir/$tmp_model_id/results"

# "--dtype bfloat16"
# "--attn_implementation flash_attention_2"

# python run_eval_b.py \
#     --model_name_or_path "$model_name_or_path" \
#     --dtype "bfloat16" \
#     --attn_implementation "sdpa" \
#     --use_pipeline "False" \
#     --dataset_file "/projects/bhuang/corpus/speech/nemo_manifests/speech-recognition-community-v2/dev_data/fr/validation/validation_speech-recognition-community-v2_dev_data_manifest.json" \
#     --audio_column_name "audio_filepath" \
#     --streaming \
#     --samples_per_dataset 1 \
#     --batch_size 16 \
#     --language "french" \
#     --task "transcribe" \
#     --return_timestamps "True" \
#     --num_beams 1 \
#     --generation_max_length 448

# short_form
eval_short_form_function() {
    test_file=$1
    # echo "test_file: $1"

    infer_opt=(
        "--model_name_or_path $model_name_or_path"
        "--dtype float16"
        "--attn_implementation sdpa"
        "--use_pipeline False"
        "--batch_size 32"
        "--language $lang"
        "--task transcribe"
        "--return_timestamps False"
        "--num_beams 1"
        "--generation_max_length 256"
    )
    decode_suffix=_greedy

    # Join array elements into a single string separated by spaces
    infer_opt_string="${infer_opt[*]}"

    test_name="${test_file##*/}"
    test_name="${test_name%.*}"
    test_name="${test_name%_manifest}"
    tmp_outdir="${outdir}_${test_name}${decode_suffix}"

    python run_eval_b.py \
        $infer_opt_string \
        --dataset_file $test_file \
        --audio_column_name "audio_filepath" \
        --streaming "True" \
        --only_short_form "True" \
        --output_file ${tmp_outdir}/predictions.json

    python scripts/compute_wer_and_alignment.py \
        --input_file_path ${tmp_outdir}/predictions.json \
        --language $lang \
        --output_dir ${tmp_outdir} 2>&1 | tee ${tmp_outdir}/log.txt
}

# chunked long form
eval_long_form_chunked_function() {
    test_file=$1
    # echo "test_file: $1"

    infer_opt=(
        "--model_name_or_path $model_name_or_path"
        "--dtype float16"
        "--attn_implementation sdpa"
        "--use_pipeline True"
        "--chunk_length_s 30"
        "--batch_size 1"
        "--language $lang"
        "--task transcribe"
        "--return_timestamps False"
        "--num_beams 1"
        "--generation_max_length 256"
    )
    decode_suffix=_greedy_chunk30

    # Join array elements into a single string separated by spaces
    infer_opt_string="${infer_opt[*]}"

    test_name="${test_file##*/}"
    test_name="${test_name%.*}"
    test_name="${test_name%_manifest}"
    tmp_outdir="${outdir}_${test_name}${decode_suffix}"

    python run_eval_b.py \
        $infer_opt_string \
        --dataset_file $test_file \
        --audio_column_name "audio_filepath" \
        --streaming "True" \
        --output_file ${tmp_outdir}/predictions.json

    python scripts/compute_wer_and_alignment.py \
        --input_file_path ${tmp_outdir}/predictions.json \
        --language $lang \
        --output_dir ${tmp_outdir} 2>&1 | tee ${tmp_outdir}/log.txt
}

# sequential long form
eval_long_form_sequential_function() {
    test_file=$1
    # echo "test_file: $1"

    infer_opt=(
        "--model_name_or_path $model_name_or_path"
        "--dtype float16"
        "--attn_implementation sdpa"
        "--use_pipeline False"
        "--batch_size 32"
        "--language $lang"
        "--task transcribe"
        "--return_timestamps True"
        "--num_beams 1"
        "--generation_max_length 256"
    )
    decode_suffix=_greedy_sequential
    # "--condition_on_prev_tokens True"
    # decode_suffix=_greedy_sequential_condonprev

    # Join array elements into a single string separated by spaces
    infer_opt_string="${infer_opt[*]}"

    test_name="${test_file##*/}"
    test_name="${test_name%.*}"
    test_name="${test_name%_manifest}"
    tmp_outdir="${outdir}_${test_name}${decode_suffix}"

    python run_eval_b.py \
        $infer_opt_string \
        --dataset_file $test_file \
        --audio_column_name "audio_filepath" \
        --streaming "True" \
        --output_file ${tmp_outdir}/predictions.json

    python scripts/compute_wer_and_alignment.py \
        --input_file_path ${tmp_outdir}/predictions.json \
        --language $lang \
        --output_dir ${tmp_outdir} 2>&1 | tee ${tmp_outdir}/log.txt
}

eval_long_form_sequential_prev_function() {
    test_file=$1
    # echo "test_file: $1"

    infer_opt=(
        "--model_name_or_path $model_name_or_path"
        "--dtype float16"
        "--attn_implementation sdpa"
        "--use_pipeline False"
        "--batch_size 32"
        "--language $lang"
        "--task transcribe"
        "--return_timestamps True"
        "--condition_on_prev_tokens True"
        "--num_beams 1"
        "--generation_max_length 256"
    )
    decode_suffix=_greedy_sequential_condonprev

    # Join array elements into a single string separated by spaces
    infer_opt_string="${infer_opt[*]}"

    test_name="${test_file##*/}"
    test_name="${test_name%.*}"
    test_name="${test_name%_manifest}"
    tmp_outdir="${outdir}_${test_name}${decode_suffix}"

    python run_eval_b.py \
        $infer_opt_string \
        --dataset_file $test_file \
        --audio_column_name "audio_filepath" \
        --streaming "True" \
        --output_file ${tmp_outdir}/predictions.json

    python scripts/compute_wer_and_alignment.py \
        --input_file_path ${tmp_outdir}/predictions.json \
        --language $lang \
        --output_dir ${tmp_outdir} 2>&1 | tee ${tmp_outdir}/log.txt
}

if [ "$lang" = "it" ]; then
    # it
    short_form_test_files=(
        "/projects/bhuang/corpus/speech/nemo_manifests/mozilla-foundation/common_voice_17_0/it/validation/validation_mozilla-foundation_common_voice_17_0_manifest.json"
        "/projects/bhuang/corpus/speech/nemo_manifests/google/fleurs/it_it/validation/validation_google_fleurs_manifest.json"
        # "/projects/bhuang/corpus/speech/nemo_manifests/mozilla-foundation/common_voice_17_0/it/test/test_mozilla-foundation_common_voice_17_0_manifest.json"
        # "/projects/bhuang/corpus/speech/nemo_manifests/facebook/multilingual_librispeech/italian/test/test_facebook_multilingual_librispeech_manifest.json"
        # "/projects/bhuang/corpus/speech/nemo_manifests/facebook/voxpopuli/it/test/test_facebook_voxpopuli_manifest.json"
        # "/projects/bhuang/corpus/speech/nemo_manifests/multilingual-tedx/it-it/test/test_mtedx_asr_manifest.json"
        # "/projects/bhuang/corpus/speech/nemo_manifests/google/fleurs/it_it/test/test_google_fleurs_manifest.json"
    )
    # long_form_test_files=(
    #     "/projects/bhuang/corpus/speech/nemo_manifests/speech-recognition-community-v2/dev_data/it/validation/validation_speech-recognition-community-v2_dev_data_manifest.json"
    #     "/projects/bhuang/corpus/speech/nemo_manifests/multilingual-tedx/it-it/test_long_form/test_mtedx_asr_manifest.json"
    # )
elif [ "$lang" = "fr" ]; then
    # fr
    short_form_test_files=(
        "/projects/bhuang/corpus/speech/nemo_manifests/mozilla-foundation/common_voice_17_0/fr/validation/validation_mozilla-foundation_common_voice_17_0_manifest.json"
        "/projects/bhuang/corpus/speech/nemo_manifests/google/fleurs/fr_fr/validation/validation_google_fleurs_manifest.json"
        # "/projects/bhuang/corpus/speech/nemo_manifests/mozilla-foundation/common_voice_17_0/fr/test/test_mozilla-foundation_common_voice_17_0_manifest.json"
        # "/projects/bhuang/corpus/speech/nemo_manifests/facebook/multilingual_librispeech/french/test/test_facebook_multilingual_librispeech_manifest.json"
        # "/projects/bhuang/corpus/speech/nemo_manifests/facebook/voxpopuli/fr/test/test_facebook_voxpopuli_manifest.json"
        # "/projects/bhuang/corpus/speech/nemo_manifests/multilingual-tedx/fr-fr/test/test_mtedx_asr_manifest.json"
        # "/projects/bhuang/corpus/speech/nemo_manifests/gigant/african_accented_french/test/test_gigant_african_accented_french_manifest.json"
        # "/projects/bhuang/corpus/speech/nemo_manifests/google/fleurs/fr_fr/test/test_google_fleurs_manifest.json"
        # # "/projects/bhuang/corpus/speech/nemo_manifests/BrunoHays/Accueil_UBS/test/test_BrunoHays_Accueil_UBS_manifest.json"
        # "/projects/bhuang/corpus/speech/zaion/hmhm_10h/test_zaion_hmhm_10h_manifest.json"
        # "/projects/bhuang/corpus/speech/zaion/carglass_5h/test_zaion_carglass_5h_manifest.json"
        # "/projects/bhuang/corpus/speech/zaion/dekuple_5h/test_zaion_dekuple_5h_manifest.json"
        # "/projects/bhuang/corpus/speech/zaion/lbpa_2.35h/test_zaion_lbpa_2h_manifest.json"
    )
    # long_form_test_files=(
    #     "/projects/bhuang/corpus/speech/nemo_manifests/speech-recognition-community-v2/dev_data/fr/validation/validation_speech-recognition-community-v2_dev_data_manifest.json"
    #     "/projects/bhuang/corpus/speech/nemo_manifests/multilingual-tedx/fr-fr/test_long_form/test_mtedx_asr_manifest.json"
    #     "/projects/bhuang/corpus/speech/zaion/dekuple_5h_merged/test_zaion_dekuple_5h_merged_by_channel_manifest.json"
    #     "/projects/bhuang/corpus/speech/zaion/dekuple_5h_merged/test_zaion_dekuple_5h_merged_by_conversation_manifest.json"
    # )
elif [ "$lang" = "es" ]; then
    # es
    short_form_test_files=(
        "/projects/bhuang/corpus/speech/nemo_manifests/mozilla-foundation/common_voice_17_0/es/validation/validation_mozilla-foundation_common_voice_17_0_manifest.json"
        "/projects/bhuang/corpus/speech/nemo_manifests/google/fleurs/es_419/validation/validation_google_fleurs_manifest.json"
        # "/projects/bhuang/corpus/speech/nemo_manifests/mozilla-foundation/common_voice_17_0/es/test/test_mozilla-foundation_common_voice_17_0_manifest.json"
        # "/projects/bhuang/corpus/speech/nemo_manifests/facebook/multilingual_librispeech/spanish/test/test_facebook_multilingual_librispeech_manifest.json"
        # "/projects/bhuang/corpus/speech/nemo_manifests/facebook/voxpopuli/es/test/test_facebook_voxpopuli_manifest.json"
        # "/projects/bhuang/corpus/speech/nemo_manifests/multilingual-tedx/es-es/test/test_mtedx_asr_manifest.json"
        # "/projects/bhuang/corpus/speech/nemo_manifests/google/fleurs/es_419/test/test_google_fleurs_manifest.json"
    )
    # long_form_test_files=(
    #     "/projects/bhuang/corpus/speech/nemo_manifests/speech-recognition-community-v2/dev_data/es/validation/validation_speech-recognition-community-v2_dev_data_manifest.json"
    #     "/projects/bhuang/corpus/speech/nemo_manifests/multilingual-tedx/es-es/test_long_form/test_mtedx_asr_manifest.json"
    # )
elif [ "$lang" = "pt" ]; then
    # pt
    short_form_test_files=(
        "/projects/bhuang/corpus/speech/nemo_manifests/mozilla-foundation/common_voice_17_0/pt/validation/validation_mozilla-foundation_common_voice_17_0_manifest.json"
        "/projects/bhuang/corpus/speech/nemo_manifests/google/fleurs/pt_br/validation/validation_google_fleurs_manifest.json"
        # "/projects/bhuang/corpus/speech/nemo_manifests/mozilla-foundation/common_voice_17_0/pt/test/test_mozilla-foundation_common_voice_17_0_manifest.json"
        # "/projects/bhuang/corpus/speech/nemo_manifests/facebook/multilingual_librispeech/portuguese/test/test_facebook_multilingual_librispeech_manifest.json"
        # "/projects/bhuang/corpus/speech/nemo_manifests/multilingual-tedx/pt-pt/test/test_mtedx_asr_manifest.json"
        # "/projects/bhuang/corpus/speech/nemo_manifests/google/fleurs/pt_br/test/test_google_fleurs_manifest.json"

    )
    # long_form_test_files=(
    #     "/projects/bhuang/corpus/speech/nemo_manifests/speech-recognition-community-v2/dev_data/pt/validation/validation_speech-recognition-community-v2_dev_data_manifest.json"
    #     "/projects/bhuang/corpus/speech/nemo_manifests/multilingual-tedx/pt-pt/test_long_form/test_mtedx_asr_manifest.json"
    # )
elif [ "$lang" = "de" ]; then
    # de
    short_form_test_files=(
        "/projects/bhuang/corpus/speech/nemo_manifests/mozilla-foundation/common_voice_17_0/de/validation/validation_mozilla-foundation_common_voice_17_0_manifest.json"
        "/projects/bhuang/corpus/speech/nemo_manifests/google/fleurs/de_de/validation/validation_google_fleurs_manifest.json"
        # "/projects/bhuang/corpus/speech/nemo_manifests/mozilla-foundation/common_voice_17_0/de/test/test_mozilla-foundation_common_voice_17_0_manifest.json"
        # "/projects/bhuang/corpus/speech/nemo_manifests/facebook/multilingual_librispeech/german/test/test_facebook_multilingual_librispeech_manifest.json"
        # "/projects/bhuang/corpus/speech/nemo_manifests/facebook/voxpopuli/de/test/test_facebook_voxpopuli_manifest.json"
        # "/projects/bhuang/corpus/speech/nemo_manifests/multilingual-tedx/de-de/test/test_mtedx_asr_manifest.json"
        # "/projects/bhuang/corpus/speech/nemo_manifests/google/fleurs/de_de/test/test_google_fleurs_manifest.json"

    )
    # long_form_test_files=(
    #     "/projects/bhuang/corpus/speech/nemo_manifests/speech-recognition-community-v2/dev_data/de/validation/validation_speech-recognition-community-v2_dev_data_manifest.json"
    #     "/projects/bhuang/corpus/speech/nemo_manifests/multilingual-tedx/de-de/test_long_form/test_mtedx_asr_manifest.json"
    # )
elif [ "$lang" = "nl" ]; then
    # nl
    short_form_test_files=(
        "/projects/bhuang/corpus/speech/nemo_manifests/mozilla-foundation/common_voice_17_0/nl/validation/validation_mozilla-foundation_common_voice_17_0_manifest.json"
        "/projects/bhuang/corpus/speech/nemo_manifests/google/fleurs/nl_nl/validation/validation_google_fleurs_manifest.json"
        # "/projects/bhuang/corpus/speech/nemo_manifests/mozilla-foundation/common_voice_17_0/nl/test/test_mozilla-foundation_common_voice_17_0_manifest.json"
        # "/projects/bhuang/corpus/speech/nemo_manifests/facebook/multilingual_librispeech/german/test/test_facebook_multilingual_librispeech_manifest.json"
        # "/projects/bhuang/corpus/speech/nemo_manifests/facebook/voxpopuli/nl/test/test_facebook_voxpopuli_manifest.json"
        # "/projects/bhuang/corpus/speech/nemo_manifests/google/fleurs/nl_nl/test/test_google_fleurs_manifest.json"
    )
    # long_form_test_files=(
    # )
elif [ "$lang" = "en" ]; then
    # en
    short_form_test_files=(
        "/projects/bhuang/corpus/speech/nemo_manifests/mozilla-foundation/common_voice_17_0/en/validation/validation_mozilla-foundation_common_voice_17_0_manifest.json"
        "/projects/bhuang/corpus/speech/nemo_manifests/google/fleurs/en_us/validation/validation_google_fleurs_manifest.json"
        # "/projects/bhuang/corpus/speech/nemo_manifests/mozilla-foundation/common_voice_17_0/en/test/test_mozilla-foundation_common_voice_17_0_manifest.json"
        # "/projects/bhuang/corpus/speech/nemo_manifests/openslr/librispeech_asr/test.clean/test.clean_openslr_librispeech_asr_manifest.json"
        # "/projects/bhuang/corpus/speech/nemo_manifests/openslr/librispeech_asr/test.other/test.other_openslr_librispeech_asr_manifest.json"
        # "/projects/bhuang/corpus/speech/nemo_manifests/facebook/voxpopuli/en/test/test_facebook_voxpopuli_manifest.json"
        # "/projects/bhuang/corpus/speech/nemo_manifests/LIUM/tedlium/release3/test/test_LIUM_tedlium_manifest.json"
        # "/projects/bhuang/corpus/speech/nemo_manifests/speechcolab/gigaspeech/test/test/test_speechcolab_gigaspeech_manifest.json"
        # "/projects/bhuang/corpus/speech/nemo_manifests/kensho/spgispeech/test/test/test_kensho_spgispeech_manifest.json"
        # "/projects/bhuang/corpus/speech/nemo_manifests/google/fleurs/en_us/test/test_google_fleurs_manifest.json"
    )
    # long_form_test_files=(
    # )
fi


# Iterate over the array and apply the function to each element
for test_file in "${short_form_test_files[@]}"; do
    eval_short_form_function "$test_file"
done

# # Iterate over the array and apply the function to each element
# for test_file in "${long_form_test_files[@]}"; do
#     eval_long_form_chunked_function "$test_file"
#     eval_long_form_sequential_function "$test_file"
#     # eval_long_form_sequential_prev_function "$test_file"
# done

echo "END TIME: $(date)"
