#!/bin/bash
#SBATCH --job-name=training_stage2_cnn_v6_dropout0.5_wd0.001_%j
#SBATCH --output=training_stage2_cnn_v6_dropout0.5_wd0.001_%j.out
# Walltime Limit: hh:mm:ss 
#SBATCH --time=12:20:00
#SBATCH --nodes=1
#SBATCH --ntasks=28 
#SBATCH --mail-type=ALL
#SBATCH --account=PAS0439
#SBATCH --gpus-per-node=1

START=$SECONDS

module load python/3.6-conda5.2
source activate  torch 

module load cuda
python  training_stage2_init_cnn.py

DURATION=$(( SECONDS - START ))

echo "Completed in $DURATION seconds."
sacct -j $SLURM_JOB_ID -o JobID,AllocTRES%50,Elapsed,CPUTime,TresUsageInTot,MaxRSS