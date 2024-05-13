#!/bin/bash
#SBATCH --job-name=metaeuk_%j
#SBATCH --output=metaeuk_%j.out
# Walltime Limit: hh:mm:ss 
#SBATCH --time=20:30:00
#SBATCH --nodes=1
#SBATCH --ntasks=40
#SBATCH --mail-type=END
#SBATCH --account=PAS0439

START=$SECONDS
source activate /fs/ess/PAS0439/MING/conda/metaeuk


#bin=$1
#metaeuk easy-predict ../cilates_fungi_classifier/results/fungi_bins/$bin   rumen_eukaryotes_protein/rumen_eukaryotes_protein.DB_clu90 ../cilates_fungi_classifier/results/metaeuk_out/bins/fungi/$bin /fs/scratch/PAS0439/Ming/tmp
file=$1
for f in $(cat $file);
do echo $f;
cd  /fs/ess/PAS0439/MING/databases/;
metaeuk easy-predict ../cilates_fungi_classifier/results/protozoa_bins/$f   rumen_eukaryotes_protein/rumen_eukaryotes_protein.DB_clu90 ../cilates_fungi_classifier/results/metaeuk_out/bins/protozoa/$f /fs/scratch/PAS0439/Ming/tmp
done

DURATION=$(( SECONDS - START ))

echo "Completed in $DURATION seconds."
