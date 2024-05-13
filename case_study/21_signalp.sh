#!/bin/bash
#SBATCH --job-name=signalp%j
#SBATCH --output=signalp%j.out
# Walltime Limit: hh:mm:ss 
#SBATCH --time=1:30:00
#SBATCH --nodes=1
#SBATCH --ntasks=10
#SBATCH --mail-type=ALL
#SBATCH --account=PAS0439

START=$SECONDS

cd /fs/ess/PAS0439/MING/cilates_fungi_classifier/results/metaeuk_out/databases

module load python/3.6-conda5.2
source activate /fs/scratch/PAS0439/Ming/conda/signalp

signalp6 --fastafile rumen_protozoa_protein_clu50_peptidase.faa --organism eukarya --output_dir ./ --format none --mode fast 



DURATION=$(( SECONDS - START ))

echo "Completed in $DURATION seconds."

