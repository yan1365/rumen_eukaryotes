#!/bin/bash
#SBATCH --job-name=ncbi_%j
#SBATCH --output=ncbi_%j.out
# Walltime Limit: hh:mm:ss 
#SBATCH --time=00:30:00
#SBATCH --nodes=1
#SBATCH --ntasks=30
#SBATCH --mail-type=ALL
#SBATCH --account=PAS0439

START=$SECONDS
# downloaded 10-2-2023

module load python/3.6-conda5.2

cd /fs/scratch/PAS0439/Ming/databases/gut_eukaryotes_classifier

#shuf test/test.csv > transformed_dataset/test_shuffled.csv
#shuf train/train.csv > transformed_dataset/train_shuffled.csv
#shuf val/val.csv > transformed_dataset/val_shuffled.csv


cd stage2
shuf test/test.csv > transformed_dataset/test_shuffled.csv
#shuf train/train.csv > transformed_dataset/train_shuffled.csv
#shuf val/val.csv > transformed_dataset/val_shuffled.csv

DURATION=$(( SECONDS - START ))

echo "Completed in $DURATION seconds."
sacct -j $SLURM_JOB_ID -o JobID,AllocTRES%50,Elapsed,CPUTime,TresUsageInTot,MaxRSS