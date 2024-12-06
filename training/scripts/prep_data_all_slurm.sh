#!/bin/bash

# pipeline for data prepration on slurm

input_file=$1
lang=$2

# lang=en

# mcv
prep_mcv() {
    local input_file=$1

    ./scripts/split_file.sh $input_file
    jobstring=$(sbatch run_pseudo_labelling_c2.slurm $input_file $lang)
    echo $jobstring
    jobid=${jobstring##* }
    sbatch --dependency=afterok:${jobid} scripts/prep_data_mcv.slurm $input_file $lang
}

# mls, voxpopuli, yodas
prep_others() {
    local input_file=$1

    output_file="${input_file/\/train\//\/train_concatenated\/}"
    # output_file="${output_file%.*}_zipped.json"
    jobstring=$(sbatch scripts/concat_asr_examples.slurm $input_file)
    echo $jobstring
    jobid=${jobstring##* }
    jobstring=$(sbatch --dependency=afterok:${jobid} run_pseudo_labelling_c2.slurm $output_file $lang)
    echo $jobstring
    jobid=${jobstring##* }
    jobstring=$(sbatch --dependency=afterok:${jobid} scripts/prep_data.slurm $output_file $lang)
    echo $jobstring
}

####################################################################################################

# prep_mcv $input_file

prep_others $input_file

####################################################################################################


# input_root=/lustre/fsn1/projects/rech/gkb/commun/corpus/speech/stt-pseudo-labeled-whisper-large-v3-multilingual

# input_files=(
    # "${input_root}/facebook/voxpopuli/en/train/train_facebook_voxpopuli_manifest.json"
    # "${input_root}/MLCommons/peoples_speech/clean/train/train_MLCommons_peoples_speech_manifest.json"
    # "${input_root}/MLCommons/peoples_speech/clean_sa/train/train_MLCommons_peoples_speech_manifest.json"
    # 2
    # "${input_root}/MLCommons/peoples_speech/default/train/train_MLCommons_peoples_speech_manifest.json"
    # "${input_root}/openslr/librispeech_asr/train/train_openslr_librispeech_asr_manifest.json"
    # "${input_root}/speechcolab/gigaspeech/l/train/train_speechcolab_gigaspeech_manifest.json"
    # 3
    # "${input_root}/LIUM/tedlium/release3/train/train_LIUM_tedlium_manifest.json"
    # 4
    # "${input_root}/espnet/yodas/en000/train/train_espnet_yodas_manifest.json"
    # "${input_root}/espnet/yodas/en001/train/train_espnet_yodas_manifest.json"
    # "${input_root}/espnet/yodas/en002/train/train_espnet_yodas_manifest.json"
    # "${input_root}/espnet/yodas/en003/train/train_espnet_yodas_manifest.json"
    # "${input_root}/espnet/yodas/en004/train/train_espnet_yodas_manifest.json"
    # "${input_root}/espnet/yodas/en005/train/train_espnet_yodas_manifest.json"
    # "${input_root}/espnet/yodas/en100/train/train_espnet_yodas_manifest.json"
    # "${input_root}/espnet/yodas/en101/train/train_espnet_yodas_manifest.json"
    # "${input_root}/espnet/yodas/en102/train/train_espnet_yodas_manifest.json"
    # "${input_root}/espnet/yodas/en103/train/train_espnet_yodas_manifest.json"
    # "${input_root}/espnet/yodas/en105/train/train_espnet_yodas_manifest.json"
    # 5
    # "${input_root}/espnet/yodas/en104/train/train_espnet_yodas_manifest.json"
    # "${input_root}/espnet/yodas/en106/train/train_espnet_yodas_manifest.json"
    # "${input_root}/espnet/yodas/en107/train/train_espnet_yodas_manifest.json"
    # "${input_root}/espnet/yodas/en108/train/train_espnet_yodas_manifest.json"
    # "${input_root}/espnet/yodas/en109/train/train_espnet_yodas_manifest.json"
    # "${input_root}/espnet/yodas/en110/train/train_espnet_yodas_manifest.json"
    # "${input_root}/espnet/yodas/en111/train/train_espnet_yodas_manifest.json"
    # "${input_root}/espnet/yodas/en112/train/train_espnet_yodas_manifest.json"
    # "${input_root}/espnet/yodas/en113/train/train_espnet_yodas_manifest.json"
    # "${input_root}/espnet/yodas/en114/train/train_espnet_yodas_manifest.json"
    # "${input_root}/espnet/yodas/en115/train/train_espnet_yodas_manifest.json"
    # "${input_root}/espnet/yodas/en116/train/train_espnet_yodas_manifest.json"
    # "${input_root}/espnet/yodas/en117/train/train_espnet_yodas_manifest.json"
    # "${input_root}/espnet/yodas/en118/train/train_espnet_yodas_manifest.json"
    # "${input_root}/espnet/yodas/en119/train/train_espnet_yodas_manifest.json"
    # "${input_root}/espnet/yodas/en120/train/train_espnet_yodas_manifest.json"
    # 6
#     "${input_root}/espnet/yodas/en121/train/train_espnet_yodas_manifest.json"
#     "${input_root}/espnet/yodas/en122/train/train_espnet_yodas_manifest.json"
#     "${input_root}/espnet/yodas/en123/train/train_espnet_yodas_manifest.json"
#     "${input_root}/espnet/yodas/en124/train/train_espnet_yodas_manifest.json"
#     "${input_root}/espnet/yodas/en125/train/train_espnet_yodas_manifest.json"
#     "${input_root}/espnet/yodas/en126/train/train_espnet_yodas_manifest.json"
#     "${input_root}/espnet/yodas/en127/train/train_espnet_yodas_manifest.json"
#     "${input_root}/distil-whisper/ami-ihm/ihm/train/train_distil-whisper_ami-ihm_manifest.json"
#     "${input_root}/distil-whisper/ami-sdm/sdm/train/train_distil-whisper_ami-sdm_manifest.json"
    # ${input_root}/MLCommons/peoples_speech/clean_sa/train/train_MLCommons_peoples_speech_manifest.json
    # ${input_root}/MLCommons/peoples_speech/dirty_sa/train/train_MLCommons_peoples_speech_manifest.json
#     ${input_root}/MLCommons/peoples_speech/dirty/train/train_MLCommons_peoples_speech_manifest.json
# )

## Iterate over the array and apply the function to each element
# for input_file in "${input_files[@]}"; do
#     prep_others "$input_file"
# done
