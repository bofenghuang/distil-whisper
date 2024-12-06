#!/bin/bash

# https://groups.oist.jp/scs/advanced-slurm#:~:text=To%20tell%20Slurm%20to%20start,you%20want%20to%20wait%20for.

# jobstring=$(sbatch j1.slurm)
# jobid=${jobstring##* }
# sbatch --dependency=afterany:${jobid} j2.slurm
# sbatch --dependency=afternotok:${jobid} 

num_jobs=10

# afterwhat=afterany
# afterwhat=afterok
afterwhat=afternotok

# script=run_distillation_c1.slurm
script=$1

# Submit initial job
jobid=$(sbatch $script | awk '{print $4}')
# echo "Submitted initial job with ID: ${jobid}"
echo -n "${jobid} "

# Submit remaining jobs with dependency on previous job's success
for i in $(seq 2 $num_jobs); do
    jobid=$(sbatch --dependency=${afterwhat}:${jobid} $script | awk '{print $4}')
    # echo "Submitted job ${i} with ID: ${jobid}, dependent on previous job success"
    echo -n "${jobid} "
done

# echo "All $num_jobs jobs have been submitted in sequence"


## c1: v3's enc + v3-turbo's de; specaugxtimex03x10x2; 922503
## c2: v3's enc + v3-turbo's de; specaugxtimex022x30x2: 922504
## c3: distil-large-v3; specaugxtimex03x10x2; 929388
## c4: whisper-large-v3-init2; specaugxtimex03x10x2; 912327
## c5: v3's enc + v3-turbo's de; specaugxtimex03x10x2; bs 2048, ep 100; 917022
# c6: v3's enc + v3-turbo's de; specaugxtimex03x10x2; new dataload impl; 917019; 953587 956224 956437 956607 957625 965547
## c7: v3's enc + v3-turbo's de; specaugxtimex03x10x2; bs 2048, ep 100, lr 1e-4; 929444
## c8: v3's enc + v3-turbo's de; specaugxtimex03x10x2; 5k + 10k (10%); 922209, 922210
### c9: v3's enc + v3-turbo's de; specaugxtimex03x10x2; 5k + 14k (20%); 
## c10: v3's enc + v3-turbo's de; specaugxtimex03x10x2; prev_text 0; 922465
## c11: cs; 941678

# c12: 5k, 5k, 3K cs; 965553

# c2: v3's enc + v3-turbo's de; specaugxtimex03x20x2; 970323
## c13: v3's enc + v3-turbo's de; specaugxtimex03x30x2; 965568
## c4: whisper-large-v3-init2; specaugxtimex022x30x2; 953501
# c14: whisper-large-v3-init2; specaugxtimex03x30x2; 953538

################################################################################

### c13: v3's enc + v3-turbo's de; specaugxtimex022x30x2; 46k multi, 23k en, 9k cs;

### c1: v3's enc + v3-turbo's de; specaugxtimex022x30x2; 46k multi, 46k en, 9k cs;
## c2: v3's enc + v3-turbo's de; specaugxtimex022x30x2; 46k multi, 46k en; 
## c3: v3's enc + v3-turbo's de; specaugxtimex03x10x2; 46k multi, 46k en, 9k cs; 
## c4: v3's enc + v3-turbo's de; specaugxtimex022x30x2; 46k multi, 46k en, 9k cs; new dataload impl;
### c2: v3's enc + v3-turbo's de; specaugxtimex022x30x2; 46k multi, 46k en, 9k cs; 40 ep;

### c14: v3's enc + v3-turbo's de; specaugxtimex022x30x2; 46k multi, 46k en, 9k cs; 4 layers; ep40;

# scale up english data
## c5: whisper-large-v3-init2; specaugxtimex022x30x2; 46k multi, 98k en, 9k cs; 
### c6: whisper-large-v3-init2; specaugxtimex03x10x2; 46k multi, 98k en, 9k cs;
# c11: whisper-large-v3-init2; specaugxtimex03x10x2; 46k multi, 98k en, 9k cs; 40 ep
## c7: whisper-large-v3-init2; specaugxtimex022x30x2; 46k multi, 98k en, 9k cs; 80 ep; 
## c12: whisper-large-v3-init2; specaugxtimex03x10x2; 46k multi, 98k en, 9k cs; bs 1024;

# scale up layers & training schedule
## c8: whisper-large-v3-init2; specaugxtimex022x30x2; 46k multi, 98k en, 9k cs; 4 layers;
# c9: whisper-large-v3-init2; specaugxtimex03x10x2; 46k multi, 98k en, 9k cs; 4 layers; 40 ep

# scale up layers & training schedule
### c10: whisper-large-v3-init2; specaugxtimex03x10x2; 46k multi, 98k en, 9k cs; 8 layers; 40 ep;

################################################################################

## c6: whisper-large-v3-init2; specaugxtimex03x10x2; 98k en; 80 ep, 4096 bs;
#! c6: whisper-large-v3-init2; specaugxtimex03x10x2; 98k en; 160 ep, 4096 bs; 1133762; 1138189; 1157079 1157080 1157081 1157082 1157083 1157084 1157085 1157086 1157087 1157088

## c3: whisper-large-v3-init2; specaugxtimex03x10x2; 46k multi, 46k en, 9k cs; 80 ep, 4096 bs; 1068922

#* c8: whisper-large-v3-init4; specaugxtimex03x10x2; 46k multi, 98k en, 9k cs; 80 ep, 4096 bs; 1121194; 1180037 1180038 1180039 1180040 1180041 1180042 1180043 1180044 1180045 1180046

## c2: whisper-large-v3-init2; specaugxtimex03x10x2; 30k multi, 30k en, 9k cs; 80 ep, 4096 bs; 1127675; 1134087; 1138451; 1140242; 1143899 1143900 1143901 
#! c5: whisper-large-v3-init2; specaugxtimex03x10x2; 30k multi, 30k en, 9k cs; 160 ep, 4096 bs; 1157053 1157054 1157055 1157056 1157057 1157058 1157059 1157060 1157061 1157062
## c7: whisper-large-v3-init2; specaugxtimex03x10x2; 30k multi, 98k en, 9k cs; 80 ep, 4096 bs; 

# ---

## c1: v3's enc + v3-turbo's de; specaugxtimex022x30x2; 46k multi, 46k en, 9k cs, 7k tr; 1086583
## c5: whisper-large-v3-init2; specaugxtimex03x10x2; 46k multi, 98k en, 9k cs, 7k tr; 1086584

# ---

#* b; de; 1168368; 1177988; 1180090
## b1; de; bs 1024; 1121564 1121565 1121566 

# ---

#* d: 1140102; 1143820; 1157090; 1180608
