#!/bin/bash
#SBATCH --job-name=OrthoFinder_%j
#SBATCH --output=OrthoFinder_%j.out
# Walltime Limit: hh:mm:ss 
#SBATCH --time=120:30:00
#SBATCH --nodes=1
#SBATCH --ntasks=40
#SBATCH --mail-type=ALL
#SBATCH --account=PAS0439

START=$SECONDS
source activate /fs/ess/PAS0439/MING/conda/OrthoFinder

cd  /fs/ess/PAS0439/MING/cilates_fungi_classifier/results/
orthofinder -f orthofinder_in -o /fs/scratch/PAS0439/Ming/tmp/orthofinder_out -d 
DURATION=$(( SECONDS - START ))

echo "Completed in $DURATION seconds."
