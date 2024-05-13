#!/bin/bash
#SBATCH --job-name=GutEuk_%j
#SBATCH --output=GutEuk_%j.out
# Walltime Limit: hh:mm:ss 
#SBATCH --time=48:30:00
#SBATCH --nodes=1
#SBATCH --ntasks=10
#SBATCH --mail-type=FAIL
#SBATCH --account=PAS0439

START=$SECONDS
source deactivate
source activate torch
file=$1
for f in $(cat $file);
do cd /fs/scratch/PAS0439/Ming/rummeta-370/co-assembly;
mkdir /fs/scratch/PAS0439/Ming/tmp/${f};
cp ${f} /fs/scratch/PAS0439/Ming/tmp/${f};
/users/PAS1855/yan1365/.conda/envs/torch/bin/python /users/PAS1855/yan1365/rumen_eukaryotes/rumen_eukaryotes_mining/GutEuk/GutEuk.py -i /fs/scratch/PAS0439/Ming/tmp/${f}/${f} -o /fs/scratch/PAS0439/Ming/tmp/${f} -t 10  --to_fasta 
cp /fs/scratch/PAS0439/Ming/tmp/${f}/*_GutEuk_eukaryotes.fasta /fs/ess/PAS0439/MING/cilates_fungi_classifier/results/eukaryotes_contigs;
rm -r /fs/scratch/PAS0439/Ming/tmp/${f};
done
DURATION=$(( SECONDS - START ))

echo "Completed in $DURATION seconds."
