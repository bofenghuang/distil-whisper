#!/bin/bash

# https://groups.oist.jp/scs/advanced-slurm#:~:text=To%20tell%20Slurm%20to%20start,you%20want%20to%20wait%20for.

# first, we submit the original job, and capture the output. 
# Slurm prints "submitted batch job <jobid>" when we submit a job.
# We store that output in jobstring using $( ... )

jobstring=$(sbatch j1.slurm)

# The last word in jobstring is the job ID. There are several ways to get it,
# but the shortest is with parameter expansion: ${jobstring##* }

jobid=${jobstring##* }

# Now submit j2.slurm as a dependant job to j1:

sbatch --dependency=afterany:${jobid} j2.slurm