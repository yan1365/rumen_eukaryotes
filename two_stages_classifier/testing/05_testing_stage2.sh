#!/bin/bash
#SBATCH --job-name=testing_%j
#SBATCH --output=testing_%j.out
# Walltime Limit: hh:mm:ss 
#SBATCH --time=6:20:00
#SBATCH --nodes=1
#SBATCH --ntasks=20 
#SBATCH --mail-type=ALL
#SBATCH --account=PAS0439
#SBATCH --gpus-per-node=1

START=$SECONDS

module load python/3.6-conda5.2
source activate  torch 

module load cuda
#python 05_testing_stage2_kmerfre.py

python 05_testing_stage2_cnn.py

DURATION=$(( SECONDS - START ))

echo "Completed in $DURATION seconds."
sacct -j $SLURM_JOB_ID -o JobID,AllocTRES%50,Elapsed,CPUTime,TresUsageInTot,MaxRSS