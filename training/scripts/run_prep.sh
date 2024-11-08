#!/bin/bash

# pipeline for data prepration

# mcv
# input_file="/lustre/fsn1/projects/rech/cjc/commun/corpus/speech/nemo_manifests/mozilla-foundation/common_voice_17_0/pt/train/train_mozilla-foundation_common_voice_17_0_manifest.json"
# output_file="${input_file%.*}_whisper_large_v3_merged.json"
# output_file="${input_file%.*}_whisper_large_v3.json"
# ./scripts/split_file.sh $input_file
# sbatch run_pseudo_labelling_c2.slurm
# ./scripts/merge_files.sh $input_file
# sbatch scripts/prep_data_mcv.slurm $output_file

# mls, voxpopuli, yodas
# input_file="/lustre/fsn1/projects/rech/cjc/commun/corpus/speech/nemo_manifests/facebook/multilingual_librispeech/portuguese/train/train_facebook_multilingual_librispeech_manifest.json"
# output_file="${input_file/\/train\//\/train_concatenated\/}"
# final_output_file="${output_file%.*}_whisper_large_v3.json"
# sbatch scripts/concat_asr_examples.slurm $input_file
# ./scripts/split_file.sh "$output_file"
# sbatch run_pseudo_labelling_c2.slurm
# ./scripts/merge_files.sh "$output_file"
# sbatch scripts/prep_data.slurm $final_output_file

########################################################################################################################################################################################################

# lang="fr"
# lang="es"
# lang="nl"
# lang="de"
lang="en"

# mcv
# input_file="/lustre/fsn1/projects/rech/cjc/commun/corpus/speech/nemo_manifests/mozilla-foundation/common_voice_17_0/pt/train/train_mozilla-foundation_common_voice_17_0_manifest.json"
# input_file="/lustre/fsn1/projects/rech/cjc/commun/corpus/speech/nemo_manifests/mozilla-foundation/common_voice_17_0/de/train/train_mozilla-foundation_common_voice_17_0_manifest.json"
# input_file="/lustre/fsn1/projects/rech/cjc/commun/corpus/speech/nemo_manifests/mozilla-foundation/common_voice_17_0/fr/train/train_mozilla-foundation_common_voice_17_0_manifest.json"
# input_file="/lustre/fsn1/projects/rech/cjc/commun/corpus/speech/nemo_manifests/gigant/african_accented_french/train/train_gigant_african_accented_french_manifest.json"
# input_file="/lustre/fsn1/projects/rech/cjc/commun/corpus/speech/nemo_manifests/mozilla-foundation/common_voice_17_0/es/train/train_mozilla-foundation_common_voice_17_0_manifest.json"
# input_file="/lustre/fsn1/projects/rech/cjc/commun/corpus/speech/nemo_manifests/mozilla-foundation/common_voice_17_0/nl/train/train_mozilla-foundation_common_voice_17_0_manifest.json"
# input_file="/lustre/fsn1/projects/rech/cjc/commun/corpus/speech/nemo_manifests/mozilla-foundation/common_voice_17_0/en/train/train_mozilla-foundation_common_voice_17_0_manifest.json"
# ./scripts/split_file.sh $input_file
# jobstring=$(sbatch run_pseudo_labelling_c2.slurm $input_file $lang)
# echo $jobstring
# jobid=${jobstring##* }
# sbatch --dependency=afterok:${jobid} scripts/prep_data_mcv.slurm $input_file

# mls, voxpopuli, yodas
# input_file="/lustre/fsn1/projects/rech/cjc/commun/corpus/speech/nemo_manifests/multilingual-tedx/it-it/train/train_asr.json"

# input_file="/lustre/fsn1/projects/rech/cjc/commun/corpus/speech/nemo_manifests/facebook/multilingual_librispeech/portuguese/train/train_facebook_multilingual_librispeech_manifest.json"
# input_file="/lustre/fsn1/projects/rech/cjc/commun/corpus/speech/nemo_manifests/multilingual-tedx/pt-pt/train/train_asr.json"
# input_file="/lustre/fsn1/projects/rech/cjc/commun/corpus/speech/nemo_manifests/espnet/yodas/pt000/train/train_espnet_yodas_manifest.json"
# input_file="/lustre/fsn1/projects/rech/cjc/commun/corpus/speech/nemo_manifests/espnet/yodas/pt100/train/train_espnet_yodas_manifest.json"
# input_file="/lustre/fsn1/projects/rech/cjc/commun/corpus/speech/nemo_manifests/espnet/yodas/pt101/train/train_espnet_yodas_manifest.json"

# input_file="/lustre/fsn1/projects/rech/cjc/commun/corpus/speech/nemo_manifests/facebook/multilingual_librispeech/german/train/train_facebook_multilingual_librispeech_manifest.json"
# input_file="/lustre/fsn1/projects/rech/cjc/commun/corpus/speech/nemo_manifests/facebook/voxpopuli/de/train/train_facebook_voxpopuli_manifest.json"
# input_file="/lustre/fsn1/projects/rech/cjc/commun/corpus/speech/nemo_manifests/multilingual-tedx/de-de/train/train_asr.json"
# input_file="/lustre/fsn1/projects/rech/cjc/commun/corpus/speech/nemo_manifests/espnet/yodas/de000/train/train_espnet_yodas_manifest.json"
# input_file="/lustre/fsn1/projects/rech/cjc/commun/corpus/speech/nemo_manifests/espnet/yodas/de100/train/train_espnet_yodas_manifest.json"
# input_file="/lustre/fsn1/projects/rech/cjc/commun/corpus/speech/nemo_manifests/espnet/yodas/de101/train/train_espnet_yodas_manifest.json"

# input_file="/lustre/fsn1/projects/rech/cjc/commun/corpus/speech/nemo_manifests/facebook/multilingual_librispeech/french/train/train_facebook_multilingual_librispeech_manifest.json"
# input_file="/lustre/fsn1/projects/rech/cjc/commun/corpus/speech/nemo_manifests/facebook/voxpopuli/fr/train/train_facebook_voxpopuli_manifest.json"
# input_file="/lustre/fsn1/projects/rech/cjc/commun/corpus/speech/nemo_manifests/multilingual-tedx/fr-fr/train/train_asr.json"
# input_file="/lustre/fsn1/projects/rech/cjc/commun/corpus/speech/nemo_manifests/espnet/yodas/fr000/train/train_espnet_yodas_manifest.json"
# input_file="/lustre/fsn1/projects/rech/cjc/commun/corpus/speech/nemo_manifests/espnet/yodas/fr100/train/train_espnet_yodas_manifest.json"
# input_file="/lustre/fsn1/projects/rech/cjc/commun/corpus/speech/nemo_manifests/espnet/yodas/fr101/train/train_espnet_yodas_manifest.json"
# input_file="/lustre/fsn1/projects/rech/cjc/commun/corpus/speech/nemo_manifests/espnet/yodas/fr102/train/train_espnet_yodas_manifest.json"
# input_file="/lustre/fsn1/projects/rech/cjc/commun/corpus/speech/nemo_manifests/espnet/yodas/fr103/train/train_espnet_yodas_manifest.json"

# input_file="/lustre/fsn1/projects/rech/cjc/commun/corpus/speech/nemo_manifests/facebook/multilingual_librispeech/spanish/train/train_facebook_multilingual_librispeech_manifest.json"
# input_file="/lustre/fsn1/projects/rech/cjc/commun/corpus/speech/nemo_manifests/facebook/voxpopuli/es/train/train_facebook_voxpopuli_manifest.json"
# input_file="/lustre/fsn1/projects/rech/cjc/commun/corpus/speech/nemo_manifests/multilingual-tedx/es-es/train/train_asr.json"
# input_file="/lustre/fsn1/projects/rech/cjc/commun/corpus/speech/nemo_manifests/espnet/yodas/es000/train/train_espnet_yodas_manifest.json"
# input_file="/lustre/fsn1/projects/rech/cjc/commun/corpus/speech/nemo_manifests/espnet/yodas/es100/train/train_espnet_yodas_manifest.json"
# input_file="/lustre/fsn1/projects/rech/cjc/commun/corpus/speech/nemo_manifests/espnet/yodas/es101/train/train_espnet_yodas_manifest.json"

# input_file="/lustre/fsn1/projects/rech/cjc/commun/corpus/speech/nemo_manifests/facebook/multilingual_librispeech/dutch/train/train_facebook_multilingual_librispeech_manifest.json"
# input_file="/lustre/fsn1/projects/rech/cjc/commun/corpus/speech/nemo_manifests/facebook/voxpopuli/nl/train/train_facebook_voxpopuli_manifest.json"
# input_file="/lustre/fsn1/projects/rech/cjc/commun/corpus/speech/nemo_manifests/espnet/yodas/nl000/train/train_espnet_yodas_manifest.json"
# input_file="/lustre/fsn1/projects/rech/cjc/commun/corpus/speech/nemo_manifests/espnet/yodas/nl100/train/train_espnet_yodas_manifest.json"

# input_file="/lustre/fsn1/projects/rech/cjc/commun/corpus/speech/nemo_manifests/facebook/voxpopuli/en/train/train_facebook_voxpopuli_manifest.json"
# input_file="/lustre/fsn1/projects/rech/cjc/commun/corpus/speech/nemo_manifests/MLCommons/peoples_speech/clean/train/train_MLCommons_peoples_speech_manifest.json"
# input_file="/lustre/fsn1/projects/rech/cjc/commun/corpus/speech/nemo_manifests/MLCommons/peoples_speech/clean_sa/train/train_MLCommons_peoples_speech_manifest.json"
# input_file="/lustre/fsn1/projects/rech/cjc/commun/corpus/speech/nemo_manifests/openslr/librispeech_asr/train.clean.100+train.clean.360+train.other.500/train.clean.100+train.clean.360+train.other.500_openslr_librispeech_asr_manifest.json"
# input_file="/lustre/fsn1/projects/rech/cjc/commun/corpus/speech/nemo_manifests/espnet/yodas/en000/train/train_espnet_yodas_manifest.json"
# input_file="/lustre/fsn1/projects/rech/cjc/commun/corpus/speech/nemo_manifests/espnet/yodas/en001/train/train_espnet_yodas_manifest.json"
# input_file="/lustre/fsn1/projects/rech/cjc/commun/corpus/speech/nemo_manifests/espnet/yodas/en002/train/train_espnet_yodas_manifest.json"
# input_file="/lustre/fsn1/projects/rech/cjc/commun/corpus/speech/nemo_manifests/espnet/yodas/en003/train/train_espnet_yodas_manifest.json"
# input_file="/lustre/fsn1/projects/rech/cjc/commun/corpus/speech/nemo_manifests/espnet/yodas/en004/train/train_espnet_yodas_manifest.json"
input_file="/lustre/fsn1/projects/rech/cjc/commun/corpus/speech/nemo_manifests/espnet/yodas/en005/train/train_espnet_yodas_manifest.json"

output_file="${input_file/\/train\//\/train_concatenated\/}"
# output_file="/lustre/fsn1/projects/rech/cjc/commun/corpus/speech/nemo_manifests/openslr/librispeech_asr/train_concatenated/train.clean.100+train.clean.360+train.other.500_openslr_librispeech_asr_manifest.json"
output_file="${output_file%.*}_zipped.json"
jobstring=$(sbatch scripts/concat_asr_examples.slurm $input_file)
echo $jobstring
jobid=${jobstring##* }
jobstring=$(sbatch --dependency=afterok:${jobid} run_pseudo_labelling_c2.slurm $output_file $lang)
echo $jobstring
jobid=${jobstring##* }
jobstring=$(sbatch --dependency=afterok:${jobid} scripts/prep_data.slurm $output_file)
echo $jobstring

########################################################################################################################################################################################################

# rm -r /lustre/fsn1/projects/rech/cjc/commun/corpus/speech/nemo_manifests/facebook/multilingual_librispeech/italian/train
# rm -r /lustre/fsn1/projects/rech/cjc/commun/corpus/speech/nemo_manifests/facebook/multilingual_librispeech/german/train
# rm -r /lustre/fsn1/projects/rech/cjc/commun/corpus/speech/nemo_manifests/facebook/multilingual_librispeech/portuguese/train
# rm -r /lustre/fsn1/projects/rech/cjc/commun/corpus/speech/nemo_manifests/facebook/voxpopuli/it/train
# rm -r /lustre/fsn1/projects/rech/cjc/commun/corpus/speech/nemo_manifests/facebook/voxpopuli/de/train

# rm -r /lustre/fsn1/projects/rech/cjc/commun/corpus/speech/nemo_manifests/multilingual-tedx/de-de/train
# rm -r /lustre/fsn1/projects/rech/cjc/commun/corpus/speech/nemo_manifests/multilingual-tedx/de-de/valid
# rm -r /lustre/fsn1/projects/rech/cjc/commun/corpus/speech/nemo_manifests/multilingual-tedx/de-de/test

# rm -r /lustre/fsn1/projects/rech/cjc/commun/corpus/speech/nemo_manifests/espnet/yodas/it000/train/000000*
# rm -r /lustre/fsn1/projects/rech/cjc/commun/corpus/speech/nemo_manifests/espnet/yodas/de000/train/000000*
# rm -r /lustre/fsn1/projects/rech/cjc/commun/corpus/speech/nemo_manifests/espnet/yodas/pt000/train/000000*
# rm -r /lustre/fsn1/projects/rech/cjc/commun/corpus/speech/nemo_manifests/espnet/yodas/pt100/train/000000*

# dir=/lustre/fsn1/projects/rech/cjc/commun/corpus/speech/nemo_manifests/multilingual-tedx/it-it
# python scripts/tmp.py ${dir}/train/train_asr.json ${dir}/train/train_asr.json
# python scripts/tmp.py ${dir}/valid/valid_asr.json ${dir}/valid/valid_asr.json
# python scripts/tmp.py ${dir}/test/test_asr.json ${dir}/test/test_asr.json
# rm -r ${dir}/docs
# rm -r ${dir}/data

# validation_files=(
# "/lustre/fsn1/projects/rech/cjc/commun/corpus/speech/nemo_manifests/mozilla-foundation/common_voice_17_0/fr/validation/validation_mozilla-foundation_common_voice_17_0_manifest.json"
# "/lustre/fsn1/projects/rech/cjc/commun/corpus/speech/nemo_manifests/google/fleurs/fr_fr/validation/validation_google_fleurs_manifest.json"
# )

# for file in "${validation_files[@]}"; do
#     echo $file
#     python scripts/insert_column.py $file $file _language fr 64
# done
