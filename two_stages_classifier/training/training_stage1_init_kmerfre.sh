#!/bin/bash
#SBATCH --job-name=training_stage1_kmrefre_dropout0.5_wd0.001_%j
#SBATCH --output=training_stage1_kmrefre_dropout0.5_wd0.001_%j.out
# Walltime Limit: hh:mm:ss 
#SBATCH --time=6:20:00
#SBATCH --nodes=1
#SBATCH --ntasks=28 
#SBATCH --mail-type=ALL
#SBATCH --account=PAS0439
#SBATCH --gpus-per-node=1

START=$SECONDS

module load python/3.6-conda5.2
source activate  torch 

module load cuda
python  training_stage1_init_kmerfre.py

DURATION=$(( SECONDS - START ))

echo "Completed in $DURATION seconds."
sacct -j $SLURM_JOB_ID -o JobID,AllocTRES%50,Elapsed,CPUTime,TresUsageInTot,MaxRSS