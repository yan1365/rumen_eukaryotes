#!/bin/bash
#SBATCH --job-name=metaeuk_%j
#SBATCH --output=metaeuk_%j.out
# Walltime Limit: hh:mm:ss 
#SBATCH --time=2:30:00
#SBATCH --nodes=1
#SBATCH --ntasks=30
#SBATCH --mail-type=ALL
#SBATCH --account=PAS0439

START=$SECONDS
source activate /fs/ess/PAS0439/MING/conda/metaeuk
f=$1
cd  /fs/ess/PAS0439/MING/databases/
metaeuk easy-predict Thea_ISMEJ/contigs_goat+cow/$f  rumen_eukaryotes_protein/rumen_eukaryotes_protein.DB_clu90 \
Thea_ISMEJ/contigs_goat+cow/${f%.fasta}.faa  /fs/scratch/PAS0439/Ming/tmp1

DURATION=$(( SECONDS - START ))

echo "Completed in $DURATION seconds."
