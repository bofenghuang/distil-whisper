#!/bin/bash

#SBATCH --job-name=touch              # nom du travail
#SBATCH --output=logs/%x/%j.out       # output file (%j = job ID)
#SBATCH --error=logs/%x/%j.out        # error file (%j = job ID)
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1           # crucial - only 1 task per dist per node!
#SBATCH --cpus-per-task=16            # number of cores per tasks
#SBATCH --time=20:00:00               # temps d execution maximum demande (HH:MM:SS)
#SBATCH --qos=qos_cpu-t3              # QoS
#SBATCH --hint=nomultithread          # we get physical cores not logical
#SBATCH --partition=cpu_p1
#SBATCH --account=cjc@cpu            # cpu accounting


module load parallel/20200822

dir=/lustre/fsn1/projects/rech/gkb/commun/corpus/speech/nemo_manifests

num_jobs=${SLURM_CPUS_PER_TASK:-8}
echo "Num of workers: $num_jobs"

# find "$dir" -type f -mtime +10 -exec touch {} +

# find "$dir" -type f -mtime +10 | while read -r file; do
#     touch $file
#     echo "Processed $file"
# done

process_file() {
    touch "$1"
    echo "Processed: $1"
}
export -f process_file

# Find files and process them in parallel
# find "$dir" -type f -mtime +10 | \
find "$dir" -type f | \
    parallel --jobs $num_jobs --line-buffer process_file

echo "Done!"
