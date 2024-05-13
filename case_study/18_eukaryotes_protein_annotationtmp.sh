#!/bin/bash
#SBATCH --job-name=eggnog_%j
#SBATCH --output=eggnog_%j.out
# Walltime Limit: hh:mm:ss 
#SBATCH --time=10:30:00
#SBATCH --nodes=1
#SBATCH --ntasks=20
#SBATCH --mail-type=ALL
#SBATCH --account=PAS0439

START=$SECONDS
source activate /fs/ess/PAS0439/MING/conda/eggnog
input=$1
export PATH=/fs/ess/PAS0439/MING/conda/eggnog/bin:"$PATH"
export EGGNOG_DATA_DIR=/fs/ess/PAS0439/MING/databases/eggnog
cd /fs/ess/PAS0439/MING/cilates_fungi_classifier/results/metaeuk_out/bins/protozoa/SAGs
emapper.py -i ${input} -o ${input%.faa} --cpu 0

DURATION=$(( SECONDS - START ))

echo "Completed in $DURATION seconds."
