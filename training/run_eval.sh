#!/usr/bin/env bash


export HF_HOME="/projects/bhuang/.cache/huggingface"

export CUDA_VISIBLE_DEVICES="4"

# model_name_or_path="openai/whisper-large-v2"
model_name_or_path="eustlb/distil-large-v3-fr"
# model_name_or_path="/projects/bhuang/models/asr/whisper/fr/whisper-large-v3-distil-fr"

# python run_eval.py \
#   --model_name_or_path "$model_name_or_path" \
#   --dtype "float16" \
#   --attn_implementation "sdpa" \
#   --use_pipeline "False" \
#   --dataset_name "mozilla-foundation/common_voice_17_0" \
#   --dataset_config_name "fr" \
#   --dataset_split_name "test" \
#   --text_column_name "sentence" \
#   --only_short_form \
#   --streaming \
#   --batch_size 32 \
#   --language "french" \
#   --task "transcribe" \
#   --return_timestamps "False" \
#   --num_beams 1 \
#   --generation_max_length 256

# transcription
#   --text_column_name "raw_transcription" \
#   --dtype "float16" \
    # --only_short_form \
    # --return_timestamps "False" \

# python run_eval.py \
#     --model_name_or_path "$model_name_or_path" \
#     --dtype "float16" \
#     --attn_implementation "sdpa" \
#     --use_pipeline "False" \
#     --dataset_name "google/fleurs" \
#     --dataset_config_name "fr_fr" \
#     --dataset_split_name "test" \
#     --text_column_name "transcription" \
#     --only_short_form \
#     --streaming "False" \
#     --batch_size 32 \
#     --language "french" \
#     --task "transcribe" \
#     --return_timestamps "False" \
#     --num_beams 1 \
#     --generation_max_length 256

python run_eval_b.py \
    --model_name_or_path "$model_name_or_path" \
    --dtype "float16" \
    --attn_implementation "sdpa" \
    --use_pipeline "False" \
    --dataset_name "google/fleurs" \
    --dataset_config_name "fr_fr" \
    --dataset_split_name "test" \
    --text_column_name "transcription" \
    --only_short_form "True" \
    --streaming "True" \
    --output_file "/home/bhuang/distil-whisper/training/outputs/tmp_output.json" \
    --batch_size 32 \
    --language "french" \
    --task "transcribe" \
    --return_timestamps "False" \
    --num_beams 1 \
    --generation_max_length 256

myscriptspath="/home/bhuang/myscripts"
export PYTHONPATH="${PYTHONPATH:-}:$myscriptspath"
export PYTHONUNBUFFERED=1

python scripts/compute_wer_and_alignment.py \
    --input_file_path /home/bhuang/distil-whisper/training/outputs/tmp_output.json \
    --language french \
    --output_dir /home/bhuang/distil-whisper/training/outputs/tmp

# long form chunked
# python run_eval.py \
#     --model_name_or_path "$model_name_or_path" \
#     --dtype "float16" \
#     --attn_implementation "sdpa" \
#     --use_pipeline "True" \
#     --chunk_length_s 30 \
#     --dataset_name "eustlb/french-long-form-test" \
#     --dataset_split_name "test" \
#     --text_column_name "sentence" \
#     --batch_size 1 \
#     --language "french" \
#     --task "transcribe" \
#     --return_timestamps "False" \
#     --num_beams 1 \
#     --generation_max_length 256

# long form sequential
# python run_eval.py \
#     --model_name_or_path "$model_name_or_path" \
#     --dtype "float32" \
#     --attn_implementation "sdpa" \
#     --use_pipeline "False" \
#     --dataset_name "eustlb/french-long-form-test" \
#     --dataset_split_name "test" \
#     --text_column_name "sentence" \
#     --streaming \
#     --batch_size 32 \
#     --language "french" \
#     --task "transcribe" \
#     --return_timestamps "True" \
#     --condition_on_prev_tokens "True" \
#     --num_beams 1 \
#     --generation_max_length 256

    # --condition_on_prev_tokens "True" \