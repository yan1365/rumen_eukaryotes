#!/bin/bash
#SBATCH --job-name=trimal_%j
#SBATCH --output=trimal_%j.out
# Walltime Limit: hh:mm:ss 
#SBATCH --time=04:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=10
#SBATCH --account=PAS0439
#SBATCH --mail-type=ALL
#SBATCH --mail-user=yan1365,yan.1365@osu.edu

START=$SECONDS

cd  /fs/ess/PAS0439/MING/cilates_fungi_classifier/results/busco_out/medium_quality_protozoa_bins/single-copy-marker/concatenated_gene_tree

module load python/3.6-conda5.2
source activate /fs/ess/PAS0439/MING/conda/trimal

for f in *.aln;
do
trimal -automated1 -in ${f}  -keepseqs -out ${f%.aln}_trimmed.aln
done 
DURATION=$(( SECONDS - START ))

echo "Completed in $DURATION seconds."
