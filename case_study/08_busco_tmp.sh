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

cd  /fs/scratch/PAS0439/Ming/databases/rumen_protozoa
busco -i ./ -m genome -l eukaryota_odb10 -c 8  --augustus  --out_path busco_out

DURATION=$(( SECONDS - START ))

echo "Completed in $DURATION seconds."
