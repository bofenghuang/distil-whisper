#!/bin/bash
# Copyright 2023  Bofeng Huang

#SBATCH --job-name=pseudo-labelling
#SBATCH --output=logs/%x/%j.out      # output file (%j = job ID)
#SBATCH --error=logs/%x/%j.err       # error file (%j = job ID)
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1          # crucial - only 1 task per dist per node!
#SBATCH --cpus-per-task=3            # number of cores per tasks
#SBATCH --gres=gpu:1                 # reserve 8 GPUs per node
#SBATCH --time=20:00:00              # maximum execution time (HH:MM:SS)
#SBATCH --qos=qos_gpu-t3             # QoS
#SBATCH --hint=nomultithread         # we get physical cores not logical
#SBATCH --partition=gpu_p2
#SBATCH --account=cjc@v100           # A100 accounting
#SBATCH --array=1-8

# Run pseudo labelling on V100s w/ job array
# http://www.idris.fr/eng/jean-zay/cpu/jean-zay-cpu-exec_jobarray-eng.html

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

# export CUDA_VISIBLE_DEVICES=$((SLURM_ARRAY_TASK_ID - 1))

# https://github.com/pytorch/audio/issues/1021#issuecomment-726915239
# export OMP_NUM_THREADS="1"

# hf env var
# export HF_HOME="/projects/bhuang/.cache/huggingface"
export TOKENIZERS_PARALLELISM="false"
# export BITSANDBYTES_NOWELCOME="1"
# export HF_HUB_ENABLE_HF_TRANSFER="1"
export HF_HUB_OFFLINE="1"
export HF_DATASETS_OFFLINE="1"
export HF_EVALUATE_OFFLINE="1"

# Set your number of GPUs here
# num_gpus=4

#   --dtype "bfloat16" \
# --max_label_length 128 \
    # --max_samples_per_split 1024 \

# CMD="accelerate launch --multi_gpu --num_processes=$num_gpus"
CMD="accelerate launch"

# model_name_or_path="openai/whisper-large-v3"
# model_name_or_path="bofenghuang/whisper-large-v3-french"
model_name_or_path="/gpfsdswork/projects/rech/cjc/commun/models/whisper/whisper-large-v3"

# input_file="/gpfsscratch/rech/cjc/commun/corpus/speech/nemo_manifests/mozilla-foundation/common_voice_17_0/it/train/train_mozilla-foundation_common_voice_17_0_manifest.json"
# input_file="/gpfsscratch/rech/cjc/commun/corpus/speech/nemo_manifests/facebook/multilingual_librispeech/italian/train_concatenated/train_facebook_multilingual_librispeech_manifest.json"
# input_file="/gpfsscratch/rech/cjc/commun/corpus/speech/nemo_manifests/facebook/voxpopuli/it/train_concatenated/train_facebook_voxpopuli_manifest.json"
# input_file="/gpfsscratch/rech/cjc/commun/corpus/speech/nemo_manifests/espnet/yodas/it000/train_concatenated/train_espnet_yodas_manifest.json"
# input_file="/gpfsscratch/rech/cjc/commun/corpus/speech/nemo_manifests/espnet/yodas/it100/train_concatenated/train_espnet_yodas_manifest.json"
# input_file="/gpfsscratch/rech/cjc/commun/corpus/speech/nemo_manifests/espnet/yodas/it101/train_concatenated/train_espnet_yodas_manifest.json"

# input_file="/gpfsscratch/rech/cjc/commun/corpus/speech/nemo_manifests/espnet/yodas/it000/train_concatenated/splitted_files/train_espnet_yodas_manifest_0${SLURM_ARRAY_TASK_ID}.json"
# input_file="/gpfsscratch/rech/cjc/commun/corpus/speech/nemo_manifests/espnet/yodas/it100/train_concatenated/splitted_files/train_espnet_yodas_manifest_0${SLURM_ARRAY_TASK_ID}.json"
input_file="/gpfsscratch/rech/cjc/commun/corpus/speech/nemo_manifests/espnet/yodas/it101/train_concatenated/splitted_files/train_espnet_yodas_manifest_0${SLURM_ARRAY_TASK_ID}.json"

# tmp_model_id="$(echo "${model_name_or_path}" | sed -e "s/-/\_/g" -e "s/[ |=/]/-/g")"
# outdir="./outputs/data/$tmp_model_id"

# tmp_model_id="$(echo "${model_name_or_path}" | sed -e "s/[ |=/-]/_/g")"
tmp_model_id="$(echo "${model_name_or_path##*/}" | sed -e "s/[ |=/-]/_/g")"
output_file="${input_file%.*}_${tmp_model_id}.json"
final_output_file="${output_file%.*}_wer.json"

$CMD run_pseudo_labelling_b.py \
    --model_name_or_path "$model_name_or_path" \
    --input_data_file "$input_file" \
    --output_dir "$outdir" \
    --output_data_file "$output_file" \
    --audio_column_name "audio_filepath" \
    --id_column_name "id" \
    --duration_column_name "duration" \
    --sort_by_duration True \
    --preprocessing_num_workers 6 \
    --pad_to_multiple_of 64 \
    --dataloader_num_workers 6 \
    --dtype "float16" \
    --attn_implementation "sdpa" \
    --per_device_eval_batch_size 64 \
    --language "it" \
    --task "transcribe" \
    --return_timestamps \
    --max_label_length 448 \
    --generation_num_beams 1

# python ./scripts/compute_wer.py \
#     --input_data_file "$output_file" \
#     --output_data_file "$final_output_file" \
#     --num_workers 40

echo "END TIME: $(date)"
