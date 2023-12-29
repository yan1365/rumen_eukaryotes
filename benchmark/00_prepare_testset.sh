#!/bin/bash
#SBATCH --job-name=chop_genome_%j
#SBATCH --output=chop_genome_%j.out
# Walltime Limit: hh:mm:ss 
#SBATCH --time=16:30:00
#SBATCH --nodes=1                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                              
#SBATCH --ntasks-per-node=1
#SBATCH --account=PAS0439
#SBATCH --mail-type=ALL

module load python/3.6-conda5.2 
source activate /fs/ess/PAS0439/MING/conda/MYENV

python  prepare_testset.py
DURATION=$(( SECONDS - START ))

echo "Completed in $DURATION seconds."
sacct -j $SLURM_JOB_ID -o JobID,AllocTRES%50,Elapsed,CPUTime,TresUsageInTot,MaxRSS
