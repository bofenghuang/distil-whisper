#!/usr/bin/env bash
# Copyright 2024  Bofeng Huang

# input_file=/gpfsscratch/rech/cjc/commun/corpus/speech/nemo_manifests/espnet/yodas/it000/train_concatenated/train_espnet_yodas_manifest.json
# input_file=/gpfsscratch/rech/cjc/commun/corpus/speech/nemo_manifests/espnet/yodas/it100/train_concatenated/train_espnet_yodas_manifest.json
# input_file=/gpfsscratch/rech/cjc/commun/corpus/speech/nemo_manifests/espnet/yodas/it101/train_concatenated/train_espnet_yodas_manifest.json

# take arg
input_file=$1

input_dir=${input_file%/*}
tmp_dir=${input_dir}/splitted_files

filename=${input_file##*/}
filename=${filename%.*}

splitted_files=${tmp_dir}/${filename}_*_whisper_large_v3.json
# output_file=${input_dir}/${filename}_whisper_large_v3_merged.json
output_file=${input_dir}/${filename}_whisper_large_v3.json

cat $splitted_files > $output_file

wc -l $input_file
wc -l $output_file
