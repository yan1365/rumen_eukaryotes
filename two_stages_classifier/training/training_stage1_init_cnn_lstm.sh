#!/bin/bash
#SBATCH --job-name=training_stage1_cnn_lstm_dropout0.2_wd0.001_%j
#SBATCH --output=training_stage1_cnn_lstm_dropout0.2_wd0.001_%j.out
# Walltime Limit: hh:mm:ss 
#SBATCH --time=24:20:00
#SBATCH --nodes=1
#SBATCH --ntasks=40 
#SBATCH --mail-type=ALL
#SBATCH --account=PAS0439
#SBATCH --gpus-per-node=1

START=$SECONDS

module load python/3.6-conda5.2
source activate  torch 

module load cuda
python  training_stage1_init_cnn_lstm.py

DURATION=$(( SECONDS - START ))

echo "Completed in $DURATION seconds."
sacct -j $SLURM_JOB_ID -o JobID,AllocTRES%50,Elapsed,CPUTime,TresUsageInTot,MaxRSS