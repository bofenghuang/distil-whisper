#!/bin/bash

# pipeline for data prepration on slurm

# input_file=$1
# lang=$2

lang=en

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

# input_file="/lustre/fsn1/projects/rech/cjc/commun/corpus/speech/nemo_manifests/mozilla-foundation/common_voice_17_0/fr/train/train_mozilla-foundation_common_voice_17_0_manifest.json"
# input_file="/lustre/fsn1/projects/rech/cjc/commun/corpus/speech/nemo_manifests/gigant/african_accented_french/train/train_gigant_african_accented_french_manifest.json"
# input_file="/lustre/fsn1/projects/rech/gkb/commun/corpus/speech/stt-pseudo-labeled-whisper-large-v3-multilingual/mozilla-foundation/common_voice_17_0/en/train/train_mozilla-foundation_common_voice_17_0_manifest.json"

# prep_mcv $input_file

input_files=(
    # "/lustre/fsn1/projects/rech/gkb/commun/corpus/speech/stt-pseudo-labeled-whisper-large-v3-multilingual/facebook/voxpopuli/en/train/train_facebook_voxpopuli_manifest.json"
    # "/lustre/fsn1/projects/rech/gkb/commun/corpus/speech/stt-pseudo-labeled-whisper-large-v3-multilingual/MLCommons/peoples_speech/clean/train/train_MLCommons_peoples_speech_manifest.json"
    # "/lustre/fsn1/projects/rech/gkb/commun/corpus/speech/stt-pseudo-labeled-whisper-large-v3-multilingual/MLCommons/peoples_speech/clean_sa/train/train_MLCommons_peoples_speech_manifest.json"
    # 2
    # "/lustre/fsn1/projects/rech/gkb/commun/corpus/speech/stt-pseudo-labeled-whisper-large-v3-multilingual/MLCommons/peoples_speech/default/train/train_MLCommons_peoples_speech_manifest.json"
    # "/lustre/fsn1/projects/rech/gkb/commun/corpus/speech/stt-pseudo-labeled-whisper-large-v3-multilingual/openslr/librispeech_asr/train/train_openslr_librispeech_asr_manifest.json"
    # "/lustre/fsn1/projects/rech/gkb/commun/corpus/speech/stt-pseudo-labeled-whisper-large-v3-multilingual/speechcolab/gigaspeech/l/train/train_speechcolab_gigaspeech_manifest.json"
    # 3
    # "/lustre/fsn1/projects/rech/gkb/commun/corpus/speech/stt-pseudo-labeled-whisper-large-v3-multilingual/LIUM/tedlium/release3/train/train_LIUM_tedlium_manifest.json"
    # 4
    # "/lustre/fsn1/projects/rech/gkb/commun/corpus/speech/stt-pseudo-labeled-whisper-large-v3-multilingual/espnet/yodas/en000/train/train_espnet_yodas_manifest.json"
    # "/lustre/fsn1/projects/rech/gkb/commun/corpus/speech/stt-pseudo-labeled-whisper-large-v3-multilingual/espnet/yodas/en001/train/train_espnet_yodas_manifest.json"
    # "/lustre/fsn1/projects/rech/gkb/commun/corpus/speech/stt-pseudo-labeled-whisper-large-v3-multilingual/espnet/yodas/en002/train/train_espnet_yodas_manifest.json"
    # "/lustre/fsn1/projects/rech/gkb/commun/corpus/speech/stt-pseudo-labeled-whisper-large-v3-multilingual/espnet/yodas/en003/train/train_espnet_yodas_manifest.json"
    # "/lustre/fsn1/projects/rech/gkb/commun/corpus/speech/stt-pseudo-labeled-whisper-large-v3-multilingual/espnet/yodas/en004/train/train_espnet_yodas_manifest.json"
    # "/lustre/fsn1/projects/rech/gkb/commun/corpus/speech/stt-pseudo-labeled-whisper-large-v3-multilingual/espnet/yodas/en005/train/train_espnet_yodas_manifest.json"
    # "/lustre/fsn1/projects/rech/gkb/commun/corpus/speech/stt-pseudo-labeled-whisper-large-v3-multilingual/espnet/yodas/en100/train/train_espnet_yodas_manifest.json"
    # "/lustre/fsn1/projects/rech/gkb/commun/corpus/speech/stt-pseudo-labeled-whisper-large-v3-multilingual/espnet/yodas/en101/train/train_espnet_yodas_manifest.json"
    # "/lustre/fsn1/projects/rech/gkb/commun/corpus/speech/stt-pseudo-labeled-whisper-large-v3-multilingual/espnet/yodas/en102/train/train_espnet_yodas_manifest.json"
    # "/lustre/fsn1/projects/rech/gkb/commun/corpus/speech/stt-pseudo-labeled-whisper-large-v3-multilingual/espnet/yodas/en103/train/train_espnet_yodas_manifest.json"
    # "/lustre/fsn1/projects/rech/gkb/commun/corpus/speech/stt-pseudo-labeled-whisper-large-v3-multilingual/espnet/yodas/en105/train/train_espnet_yodas_manifest.json"
    # 5
    # "/lustre/fsn1/projects/rech/gkb/commun/corpus/speech/stt-pseudo-labeled-whisper-large-v3-multilingual/espnet/yodas/en104/train/train_espnet_yodas_manifest.json"
    # "/lustre/fsn1/projects/rech/gkb/commun/corpus/speech/stt-pseudo-labeled-whisper-large-v3-multilingual/espnet/yodas/en106/train/train_espnet_yodas_manifest.json"
    # "/lustre/fsn1/projects/rech/gkb/commun/corpus/speech/stt-pseudo-labeled-whisper-large-v3-multilingual/espnet/yodas/en107/train/train_espnet_yodas_manifest.json"
    # "/lustre/fsn1/projects/rech/gkb/commun/corpus/speech/stt-pseudo-labeled-whisper-large-v3-multilingual/espnet/yodas/en108/train/train_espnet_yodas_manifest.json"
    # "/lustre/fsn1/projects/rech/gkb/commun/corpus/speech/stt-pseudo-labeled-whisper-large-v3-multilingual/espnet/yodas/en109/train/train_espnet_yodas_manifest.json"
    # "/lustre/fsn1/projects/rech/gkb/commun/corpus/speech/stt-pseudo-labeled-whisper-large-v3-multilingual/espnet/yodas/en110/train/train_espnet_yodas_manifest.json"
    # "/lustre/fsn1/projects/rech/gkb/commun/corpus/speech/stt-pseudo-labeled-whisper-large-v3-multilingual/espnet/yodas/en111/train/train_espnet_yodas_manifest.json"
    # "/lustre/fsn1/projects/rech/gkb/commun/corpus/speech/stt-pseudo-labeled-whisper-large-v3-multilingual/espnet/yodas/en112/train/train_espnet_yodas_manifest.json"
    # "/lustre/fsn1/projects/rech/gkb/commun/corpus/speech/stt-pseudo-labeled-whisper-large-v3-multilingual/espnet/yodas/en113/train/train_espnet_yodas_manifest.json"
    # "/lustre/fsn1/projects/rech/gkb/commun/corpus/speech/stt-pseudo-labeled-whisper-large-v3-multilingual/espnet/yodas/en114/train/train_espnet_yodas_manifest.json"
    # "/lustre/fsn1/projects/rech/gkb/commun/corpus/speech/stt-pseudo-labeled-whisper-large-v3-multilingual/espnet/yodas/en115/train/train_espnet_yodas_manifest.json"
    # "/lustre/fsn1/projects/rech/gkb/commun/corpus/speech/stt-pseudo-labeled-whisper-large-v3-multilingual/espnet/yodas/en116/train/train_espnet_yodas_manifest.json"
    # "/lustre/fsn1/projects/rech/gkb/commun/corpus/speech/stt-pseudo-labeled-whisper-large-v3-multilingual/espnet/yodas/en117/train/train_espnet_yodas_manifest.json"
    # "/lustre/fsn1/projects/rech/gkb/commun/corpus/speech/stt-pseudo-labeled-whisper-large-v3-multilingual/espnet/yodas/en118/train/train_espnet_yodas_manifest.json"
    # "/lustre/fsn1/projects/rech/gkb/commun/corpus/speech/stt-pseudo-labeled-whisper-large-v3-multilingual/espnet/yodas/en119/train/train_espnet_yodas_manifest.json"
    # "/lustre/fsn1/projects/rech/gkb/commun/corpus/speech/stt-pseudo-labeled-whisper-large-v3-multilingual/espnet/yodas/en120/train/train_espnet_yodas_manifest.json"
    # 6
    "/lustre/fsn1/projects/rech/gkb/commun/corpus/speech/stt-pseudo-labeled-whisper-large-v3-multilingual/espnet/yodas/en121/train/train_espnet_yodas_manifest.json"
    "/lustre/fsn1/projects/rech/gkb/commun/corpus/speech/stt-pseudo-labeled-whisper-large-v3-multilingual/espnet/yodas/en122/train/train_espnet_yodas_manifest.json"
    "/lustre/fsn1/projects/rech/gkb/commun/corpus/speech/stt-pseudo-labeled-whisper-large-v3-multilingual/espnet/yodas/en123/train/train_espnet_yodas_manifest.json"
    "/lustre/fsn1/projects/rech/gkb/commun/corpus/speech/stt-pseudo-labeled-whisper-large-v3-multilingual/espnet/yodas/en124/train/train_espnet_yodas_manifest.json"
    "/lustre/fsn1/projects/rech/gkb/commun/corpus/speech/stt-pseudo-labeled-whisper-large-v3-multilingual/espnet/yodas/en125/train/train_espnet_yodas_manifest.json"
    "/lustre/fsn1/projects/rech/gkb/commun/corpus/speech/stt-pseudo-labeled-whisper-large-v3-multilingual/espnet/yodas/en126/train/train_espnet_yodas_manifest.json"
    "/lustre/fsn1/projects/rech/gkb/commun/corpus/speech/stt-pseudo-labeled-whisper-large-v3-multilingual/espnet/yodas/en127/train/train_espnet_yodas_manifest.json"
    "/lustre/fsn1/projects/rech/gkb/commun/corpus/speech/stt-pseudo-labeled-whisper-large-v3-multilingual/distil-whisper/ami-ihm/ihm/train/train_distil-whisper_ami-ihm_manifest.json"
    "/lustre/fsn1/projects/rech/gkb/commun/corpus/speech/stt-pseudo-labeled-whisper-large-v3-multilingual/distil-whisper/ami-sdm/sdm/train/train_distil-whisper_ami-sdm_manifest.json"

)

# Iterate over the array and apply the function to each element
for input_file in "${input_files[@]}"; do
    prep_others "$input_file"
done

########################################################################################################################################################################################################

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
