#!/bin/bash
#SBATCH --job-name=tiara_%j
#SBATCH --output=tiara_%j.out
# Walltime Limit: hh:mm:ss 
#SBATCH --time=12:30:00
#SBATCH --nodes=1                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                              
#SBATCH --ntasks-per-node=20 
#SBATCH --account=PAS0439
#SBATCH --mail-type=ALL

module load python/3.6-conda5.2 
source activate /fs/scratch/PAS0439/Ming/conda/tiara

cd /fs/ess/PAS0439/MING/cilates_fungi_classifier/outputs/downstream_analysis/benchmark/

tiara -t 20 -i /fs/scratch/PAS0439/Ming/databases/gut_eukaryotes_classifier/test/test.fasta  -o test_5kb_tiara.txt


DURATION=$(( SECONDS - START ))

echo "Completed in $DURATION seconds."
sacct -j $SLURM_JOB_ID -o JobID,AllocTRES%50,Elapsed,CPUTime,TresUsageInTot,MaxRSS
