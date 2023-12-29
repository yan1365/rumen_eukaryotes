#!/bin/bash
#SBATCH --job-name=preprocess_%j
#SBATCH --output=preprocess_%j.out
# Walltime Limit: hh:mm:ss 
#SBATCH --time=00:30:00
#SBATCH --nodes=1                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                              
#SBATCH --ntasks-per-node=10
#SBATCH --account=PAS0439
#SBATCH --mail-type=ALL

module load python/3.6-conda5.2 
source activate /fs/ess/PAS0439/MING/conda/MYENV

python  ./transform.py
python ./preprocessing.py /fs/ess/PAS0439/MING/cilates_fungi_classifier/outputs/ciliates_sags_5kb.csv  /fs/ess/PAS0439/MING/cilates_fungi_classifier/outputs/ciliates_sags_npz
#python ./preprocessing.py /fs/ess/PAS0439/MING/cilates_fungi_classifier/outputs/ruminant_fungi_5kb.csv  /fs/ess/PAS0439/MING/cilates_fungi_classifier/outputs/ruminant_fungi_npz

DURATION=$(( SECONDS - START ))

echo "Completed in $DURATION seconds."
sacct -j $SLURM_JOB_ID -o JobID,AllocTRES%50,Elapsed,CPUTime,TresUsageInTot,MaxRSS
