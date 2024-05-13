#!/bin/bash
#SBATCH --job-name=GutEuk_%j
#SBATCH --output=GutEuk_%j.out
# Walltime Limit: hh:mm:ss 
#SBATCH --time=00:30:00
#SBATCH --nodes=1
#SBATCH --ntasks=10
#SBATCH --mail-type=FAIL
#SBATCH --account=PAS0439

START=$SECONDS
source deactivate
source activate torch
f=$1
mkdir /fs/scratch/PAS0439/Ming/tmp/${f};

/users/PAS1855/yan1365/.conda/envs/torch/bin/python /users/PAS1855/yan1365/rumen_eukaryotes/rumen_eukaryotes_mining/GutEuk/GutEuk.py \
-i /fs/ess/PAS0439/MING/databases/Thea_ISMEJ/assembly/${f} -o /fs/scratch/PAS0439/Ming/tmp/${f} -t 10  --to_fasta -s2 0.8 -m 15000 
cp /fs/scratch/PAS0439/Ming/tmp/${f}/*.fasta /fs/ess/PAS0439/MING/databases/Thea_ISMEJ/eukaryotic_contigs/;
rm -r /fs/scratch/PAS0439/Ming/tmp/${f};

DURATION=$(( SECONDS - START ))

echo "Completed in $DURATION seconds."
