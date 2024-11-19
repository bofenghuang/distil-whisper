#!/usr/bin/env bash
# Copyright 2023  Bofeng Huang

# prep data

set -e

echo "START TIME: $(date)"

# https://github.com/pytorch/audio/issues/1021#issuecomment-726915239
# export OMP_NUM_THREADS="1"

# cuda
# export CUDA_VISIBLE_DEVICES=""

# hf
export HF_HOME="/projects/bhuang/.cache/huggingface"
export TOKENIZERS_PARALLELISM="false"
# export BITSANDBYTES_NOWELCOME="1"
# export HF_HUB_ENABLE_HF_TRANSFER="1"
# export HF_HUB_OFFLINE="1"
# export HF_DATASETS_OFFLINE="1"
# export HF_EVALUATE_OFFLINE="1"

export CUDA_LAUNCH_BLOCKING="1"

# CPUs
num_workers=64

# take arg
input_file=$1
lang=${2:-en}
n_splits=${3:-4}
gpuindex=${4:-0}

stage=0

log_root=logs

filename=${input_file##*/}
filename=${filename%.*}
input_dir=${input_file%/*}

# create a timestamp for this run
timestamp=$(date +"%Y%m%d_%H%M%S")
log_dir="$log_root/run_$timestamp"
mkdir -p "$log_dir"

# concat
output_file="${input_file/\/train\//\/train_concatenated\/}"
# output_file="${input_file/\/projects\//\/rd_storage2\/}"

if [ $stage -le 0 ]; then
    echo -e "\n\nConcatenating examples"
    python scripts/concat_asr_examples.py \
        --input_file_path $input_file \
        --output_file_path $output_file \
        --preprocessing_batch_size 1000 \
        --num_workers $num_workers

    # tmp: remove unconcatenated audio files
    if [ "$lang" != "fr" ]; then
        echo -e "\n\nRemoving original unconcatenated audio files"
        find $input_dir -mindepth 1 -type f -name "*.wav" -delete
        find $input_dir -mindepth 1 -type d -empty -delete
    fi
fi

input_file="${output_file}"

# split input file
tmp_dir=${input_file%/*}/splitted_files

if [ $stage -le 1 ]; then
    echo -e "\n\nSplitting input file into $n_splits"
    [ -d "$tmp_dir" ] || mkdir -p "$tmp_dir"
    split -n l/$n_splits --numeric-suffixes=0 --additional-suffix=.json "$input_file" "${tmp_dir}/${filename}_"
fi

# launch inference in prallel
if [ $stage -le 2 ]; then
    echo -e "\n\nRunning inference"
    for ((i=0; i<n_splits; i++)); do
        split_file="${tmp_dir}/${filename}_0${i}.json"
        process_log="$log_dir/process${i}.log"
        # CUDA_VISIBLE_DEVICES=$i ./run_pseudo_labelling_b.sh $split_file $lang > $process_log 2>&1 &
        CUDA_VISIBLE_DEVICES=$((i+gpuindex)) ./run_pseudo_labelling_b.sh $split_file $lang > $process_log 2>&1 &
        pids+=($!)
        echo "Launched inference on GPU $i with PID ${pids[-1]}; See log in $process_log"
    done

    # wait for all background processes to complete
    # wait

    # wait for all processes and capture exit codes
    for pid in "${pids[@]}"; do
        wait $pid
        exit_codes+=($?)
    done

    # check if any process failed
    failed=0
    for i in "${!exit_codes[@]}"; do
        if [ ${exit_codes[$i]} -ne 0 ]; then
            echo "Error: Process on GPU $i (PID: ${pids[$i]}) failed with exit code ${exit_codes[$i]}"
            echo "Log file: $log_dir/gpu${i}.log"
            failed=1
        fi
    done

    # exit with error if any process failed
    if [ $failed -eq 1 ]; then
        echo "One or more processes failed. Check the logs for details."
        exit 1
    fi
fi

# merge input files
output_file=${tmp_dir%/*}/${filename}_whisper_large_v3.json

if [ $stage -le 3 ]; then
    splitted_files=${tmp_dir}/${filename}_*_whisper_large_v3.json
    echo -e "\n\nMerging infered files"
    cat $splitted_files > $output_file

    wc -l $input_file
    wc -l $output_file
    # rm -r $tmp_dir
fi

input_file="${output_file}"

# normalize (timestamps)
if [ $stage -le 4 ]; then
    echo -e "\n\nNormalizing transcripts"
    python scripts/norm_whisper_transcript.py \
        --input_file_path "$input_file" \
        --output_file_path "${input_file%.*}_norm.json" \
        --num_workers $num_workers
fi

# update prev_whisper_transcript
if [ $stage -le 5 ]; then
    echo -e "\n\nUpdate prev"
    python scripts/update_prev_whisper_transcript.py \
        --input_file_path "${input_file%.*}_norm.json" \
        --output_file_path "${input_file%.*}_norm_upprev.json" \
        --num_workers $num_workers
fi

# wer
if [ $stage -le 6 ]; then
    echo -e "\n\nComputing WER"
    python scripts/compute_wer.py \
        --input_file_path "${input_file%.*}_norm_upprev.json" \
        --output_file_path "${input_file%.*}_norm_upprev_wer.json" \
        --language $lang \
        --num_workers $num_workers
fi

# filter (upper-case, wer)
if [ $stage -le 7 ]; then
    echo -e "\n\nFiltering examples"
    python scripts/filter_whisper_transcript.py \
        --input_file_path "${input_file%.*}_norm_upprev_wer.json" \
        --output_file_path "${input_file%.*}_norm_upprev_wer_filt.json" \
        --wer_threshold 20 \
        --num_workers $num_workers
fi

# deleting audio files
if [ $stage -le 8 ]; then
    echo -e "\n\nCleaning up audio files"
    python scripts/cleanup_audio_files.py \
        --input_file_a "${input_file}" \
        --input_file_b "${input_file%.*}_norm_upprev_wer_filt.json" \
        --num_workers $num_workers
fi

# zipping audio files
# if [ $stage -le 9 ]; then
#     echo -e "\n\nZipping audio files"
#     python scripts/zip_audio_files.py \
#         --input_file_path "${input_file%.*}_norm_upprev_wer_filt.json" \
#         --output_file_path "${input_file%.*}_norm_upprev_wer_filt_zipped.json" \
#         --num_workers $num_workers
# fi

echo "END TIME: $(date)"
