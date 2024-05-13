#!/bin/bash
#SBATCH --job-name=busco_%j
#SBATCH --output=busco_%j.out
# Walltime Limit: hh:mm:ss 
#SBATCH --time=16:30:00
#SBATCH --nodes=1
#SBATCH --ntasks=8
#SBATCH --mail-type=ALL
#SBATCH --account=PAS0439

START=$SECONDS
source activate busco

cd  /fs/ess/PAS0439/MING/cilates_fungi_classifier/results/
#busco -i protozoa_bins -m genome -l alveolata_odb10 -c 8  --augustus  --out_path busco_out/protoza
#busco -i undetermined_eukaryotes_bins -m genome --auto-lineage-euk -c 8  --augustus  --out_path busco_out/undetermined_eukaryotes_bins
busco -i orthofinder_in -m genome -l alveolata_odb10 -c 8  --augustus  --out_path busco_out/medium_quality_protozoa_bins
DURATION=$(( SECONDS - START ))

echo "Completed in $DURATION seconds."
