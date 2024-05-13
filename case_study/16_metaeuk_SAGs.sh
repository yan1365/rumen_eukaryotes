#!/bin/bash
#SBATCH --job-name=metaeuk_%j
#SBATCH --output=metaeuk_%j.out
# Walltime Limit: hh:mm:ss 
#SBATCH --time=4:30:00
#SBATCH --nodes=1
#SBATCH --ntasks=20
#SBATCH --mail-type=END
#SBATCH --account=PAS0439

START=$SECONDS
source activate /fs/ess/PAS0439/MING/conda/metaeuk

cd /fs/ess/PAS0439/MING/databases/
bin=$1
metaeuk easy-predict ciliates_SAGs/high_quality/$bin   rumen_eukaryotes_protein/rumen_eukaryotes_protein.DB_clu90 ../cilates_fungi_classifier/results/metaeuk_out/bins/protozoa/SAGs/$bin /fs/scratch/PAS0439/Ming/tmp/$bin


DURATION=$(( SECONDS - START ))

echo "Completed in $DURATION seconds."
