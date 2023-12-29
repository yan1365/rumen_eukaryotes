#!/bin/bash
#SBATCH --job-name=eukrep_%j
#SBATCH --output=eukrep_%j.out
# Walltime Limit: hh:mm:ss 
#SBATCH --time=8:30:00
#SBATCH --nodes=1                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                              
#SBATCH --ntasks-per-node=20
#SBATCH --account=PAS0439
#SBATCH --mail-type=ALL

module load python/3.6-conda5.2 
source activate /fs/scratch/PAS0439/Ming/conda/eukrep-env


cd  /fs/ess/PAS0439/MING/cilates_fungi_classifier/outputs/downstream_analysis/benchmark/

EukRep -i /fs/scratch/PAS0439/Ming/databases/gut_eukaryotes_classifier/test/test.fasta -o test_eukrep_euk.fa --prokarya test_5kb_eukrep_pro.fa

DURATION=$(( SECONDS - START ))

echo "Completed in $DURATION seconds."
sacct -j $SLURM_JOB_ID -o JobID,AllocTRES%50,Elapsed,CPUTime,TresUsageInTot,MaxRSS
