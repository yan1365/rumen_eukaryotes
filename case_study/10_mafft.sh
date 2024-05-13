#!/bin/bash
#SBATCH --job-name=msa_%j
#SBATCH --output=msa_%j.out
# Walltime Limit: hh:mm:ss 
#SBATCH --time=01:10:00
#SBATCH --nodes=1
#SBATCH --ntasks=8
#SBATCH --account=PAS0439
#SBATCH --mail-type=ALL
#SBATCH --mail-user=yan1365,yan.1365@osu.edu

START=$SECONDS

cd /fs/ess/PAS0439/MING/cilates_fungi_classifier/results/busco_out/medium_quality_protozoa_bins/single-copy-marker/concatenated_gene_tree/

module load python/3.6-conda5.2
source activate /fs/ess/PAS0439/MING/conda/mafft
for f in *_concatenated.faa;
do mafft  $f   > ${f%.faa}_mafft.aln;
done

DURATION=$(( SECONDS - START ))

echo "Completed in $DURATION seconds."
