#!/bin/bash
#SBATCH --job-name=save_npz_val_%j
#SBATCH --output=save_npz_val_%j.out
# Walltime Limit: hh:mm:ss 
#SBATCH --time=4:30:00
#SBATCH --nodes=1
#SBATCH --ntasks=40 --partition=hugemem
#SBATCH --mail-type=ALL
#SBATCH --account=PAS0439

START=$SECONDS
module load miniconda3
source activate torch


./save_processed_tensor.py 128 /fs/scratch/PAS0439/Ming/databases/gut_eukaryotes_classifier/stage2/transformed_dataset/val_shuffled.csv  /fs/scratch/PAS0439/Ming/databases/gut_eukaryotes_classifier/stage2/transformed_dataset/val  


DURATION=$(( SECONDS - START ))

echo "Completed in $DURATION seconds."
sacct -j $SLURM_JOB_ID -o JobID,AllocTRES%50,Elapsed,CPUTime,TresUsageInTot,MaxRSS