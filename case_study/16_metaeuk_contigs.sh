#!/bin/bash
#SBATCH --job-name=metaeuk_%j
#SBATCH --output=metaeuk_%j.out
# Walltime Limit: hh:mm:ss 
#SBATCH --time=10:30:00
#SBATCH --nodes=1
#SBATCH --ntasks=40
#SBATCH --mail-type=ALL
#SBATCH --account=PAS0439

START=$SECONDS
source activate /fs/ess/PAS0439/MING/conda/metaeuk
f=$1
cd  /fs/ess/PAS0439/MING/databases/
#metaeuk easy-predict ../cilates_fungi_classifier/results/eukaryotes_contigs/$f  rumen_eukaryotes_protein/rumen_eukaryotes_protein.DB_clu90 ../cilates_fungi_classifier/results/metaeuk_out/contigs/${f%.fasta}_metaeuk_out.faa  /fs/scratch/PAS0439/Ming/tmp
metaeuk easy-predict ../cilates_fungi_classifier/results/eukaryotes_contigs/$f  UniProtKB_Swiss-Prot/Swiss-prot ../cilates_fungi_classifier/results/metaeuk_out/contigs/${f%.fasta}_metaeuk_out.faa  /fs/scratch/PAS0439/Ming/tmp


DURATION=$(( SECONDS - START ))

echo "Completed in $DURATION seconds."
