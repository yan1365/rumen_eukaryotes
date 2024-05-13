#!/bin/bash
#SBATCH --job-name=iqtree_%j
#SBATCH --output=iqtree_%j.out
# Walltime Limit: hh:mm:ss 
#SBATCH --time=24:00:00 
#SBATCH --nodes=1
#SBATCH --ntasks=40
#SBATCH --account=PAS0439
#SBATCH --mail-type=ALL
#SBATCH --mail-user=yan1365,yan.1365@osu.edu

START=$SECONDS

cd  /fs/ess/PAS0439/MING/cilates_fungi_classifier/results/busco_out/medium_quality_protozoa_bins/single-copy-marker/concatenated_gene_tree/iqtree

module load python/3.6-conda5.2
source activate /fs/ess/PAS0439/MING/conda/iqtree

iqtree -s busco_single_copy_marker_concatenated_msa.aln -bb 1000 -m MFP -mrate E,I,G,I+G -mfreq FU -nt AUTO
DURATION=$(( SECONDS - START ))

echo "Completed in $DURATION seconds."
