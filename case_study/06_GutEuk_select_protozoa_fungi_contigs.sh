#!/bin/bash
#SBATCH --job-name=GutEuk_%j
#SBATCH --output=GutEuk_%j.out
# Walltime Limit: hh:mm:ss 
#SBATCH --time=1:30:00
#SBATCH --nodes=1
#SBATCH --ntasks=10
#SBATCH --mail-type=ALL
#SBATCH --account=PAS0439

START=$SECONDS
source activate torch

# furtuer filter the identified Eukaryotes contigs based on the latest updates (threshold) and benchmarking 
file=$1
cd /fs/ess/PAS0439/MING/cilates_fungi_classifier/results/eukaryotes_contigs
mkdir /fs/scratch/PAS0439/Ming/tmp/${file};
cp ${file} /fs/scratch/PAS0439/Ming/tmp/${file}
/users/PAS1855/yan1365/.conda/envs/torch/bin/python /users/PAS1855/yan1365/rumen_eukaryotes/rumen_eukaryotes_mining/GutEuk/GutEuk.py -i /fs/scratch/PAS0439/Ming/tmp/${file}/${file} -o /fs/scratch/PAS0439/Ming/tmp/${file} -t 10  
cp /fs/scratch/PAS0439/Ming/tmp/${file}/*csv /fs/ess/PAS0439/MING/cilates_fungi_classifier/results/eukaryotes_contigs;
rm -r /fs/scratch/PAS0439/Ming/tmp/${file};

DURATION=$(( SECONDS - START ))

echo "Completed in $DURATION seconds."
