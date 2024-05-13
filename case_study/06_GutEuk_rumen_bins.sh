#!/bin/bash
#SBATCH --job-name=GutEuk_%j
#SBATCH --output=GutEuk_%j.out
# Walltime Limit: hh:mm:ss 
#SBATCH --time=18:30:00
#SBATCH --nodes=1
#SBATCH --ntasks=10
#SBATCH --mail-type=ALL
#SBATCH --account=PAS0439

START=$SECONDS
source deactivate
source activate torch
dir=$1

cp -r /fs/ess/PAS0439/MING/cilates_fungi_classifier/$dir /fs/scratch/PAS0439/Ming/tmp
mkdir /fs/scratch/PAS0439/Ming/tmp/$dir/tmp
/users/PAS1855/yan1365/.conda/envs/torch/bin/python /users/PAS1855/yan1365/rumen_eukaryotes/rumen_eukaryotes_mining/GutEuk/GutEuk.py -i /fs/scratch/PAS0439/Ming/tmp/$dir -o /fs/scratch/PAS0439/Ming/tmp/$dir/tmp -t 10 -b 
cp  /fs/scratch/PAS0439/Ming/tmp/$dir/tmp/*.csv /fs/ess/PAS0439/MING/cilates_fungi_classifier/results
rm -r /fs/scratch/PAS0439/Ming/tmp/$dir
DURATION=$(( SECONDS - START ))

echo "Completed in $DURATION seconds."
