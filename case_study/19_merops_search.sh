#!/bin/bash
#SBATCH --job-name=merops_%j
#SBATCH --output=merops_%j.out
# Walltime Limit: hh:mm:ss 
#SBATCH --time=10:30:00
#SBATCH --nodes=1
#SBATCH --ntasks=8
#SBATCH --mail-type=ALL
#SBATCH --account=PAS0439

START=$SECONDS
source activate checkv
input=$1
cd /fs/ess/PAS0439/MING/cilates_fungi_classifier/results/metaeuk_out/databases/
#diamond makedb --in /fs/ess/PAS0439/MING/databases/merops/merops_pepunit.lib -d /fs/ess/PAS0439/MING/databases/merops/MEROPS.db
# running a search in blastp mode
diamond blastp -d /fs/ess/PAS0439/MING/databases/merops/MEROPS.db -q $input -o ${input%.faa}_diamond_out -k 1

DURATION=$(( SECONDS - START ))

echo "Completed in $DURATION seconds."
