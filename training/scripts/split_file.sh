#!/usr/bin/env bash
# Copyright 2024  Bofeng Huang

# input_file=/gpfsscratch/rech/cjc/commun/corpus/speech/nemo_manifests/espnet/yodas/it000/train_concatenated/train_espnet_yodas_manifest.json
# input_file=/gpfsscratch/rech/cjc/commun/corpus/speech/nemo_manifests/espnet/yodas/it100/train_concatenated/train_espnet_yodas_manifest.json
# input_file=/gpfsscratch/rech/cjc/commun/corpus/speech/nemo_manifests/espnet/yodas/it101/train_concatenated/train_espnet_yodas_manifest.json

# take arg
input_file=$1

N=8

tmp_dir=${input_file%/*}/splitted_files

split -n l/$N --numeric-suffixes=1 --additional-suffix=.json $input_file ${input_file%.*}_

[ -d tmp_dir ] || mkdir $tmp_dir

mv ${input_file%.*}_*.json $tmp_dir
