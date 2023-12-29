#!/bin/bash
#SBATCH --job-name=preprocessing_test_%j
#SBATCH --output=preprocessing_test_%j.out
# Walltime Limit: hh:mm:ss 
#SBATCH --time=6:30:00
#SBATCH --nodes=1
#SBATCH --ntasks=20
#SBATCH --mail-type=ALL
#SBATCH --account=PAS0439

START=$SECONDS

module load python/3.6-conda5.2
source activate  /fs/ess/PAS0439/MING/conda/MYENV

python fasta2int_test.py

DURATION=$(( SECONDS - START ))

echo "Completed in $DURATION seconds."
sacct -j $SLURM_JOB_ID -o JobID,AllocTRES%50,Elapsed,CPUTime,TresUsageInTot,MaxRSS