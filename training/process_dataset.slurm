#!/bin/bash
# Copyright 2023  Bofeng Huang

#SBATCH --job-name=proc-data
#SBATCH --output=logs/%x/%j.out # output file (%j = job ID)
#SBATCH --error=logs/%x/%j.err # error file (%j = job ID)
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1          # crucial - only 1 task per dist per node!
#SBATCH --cpus-per-task=20         # number of cores per tasks
#SBATCH --time 01:00:00              # maximum execution time (HH:MM:SS)
#SBATCH --hint=nomultithread         # we get physical cores not logical
#SBATCH --partition=cpu_p1
#SBATCH --account=cjc@cpu


set -x -e

# module purge
# module load git-lfs
# module load unrar

workdir="/gpfsscratch/rech/cjc/commun/corpus/speech"
dirname="nemo_manifests"

cd $workdir

# tar -zcvf ${dirname}.tar.gz ${dirname}

echo "wtf"
tar -ztf ${dirname}.tar.gz  | grep test_asr_mcv13_manifest_normalized_pnc.json
# tar -zxvf ${dirname}.tar.gz ${dirname}/final/2023-11-28/test_asr_mcv13_manifest_normalized_pnc.json
