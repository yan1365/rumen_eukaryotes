#!/bin/bash
#SBATCH --job-name=eukrep_%j
#SBATCH --output=eukrep_%j.out
# Walltime Limit: hh:mm:ss 
#SBATCH --time=2:30:00
#SBATCH --nodes=1                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                              
#SBATCH --ntasks-per-node=20
#SBATCH --account=PAS0439
#SBATCH --mail-type=ALL

module load python/3.6-conda5.2 
source activate /fs/scratch/PAS0439/Ming/conda/eukrep-env


cd  /fs/ess/PAS0439/MING/cilates_fungi_classifier/outputs/downstream_analysis/benchmark/

EukRep -i /fs/ess/PAS0439/MING/databases/ciliates_SAGs/telomere_capped/telomere_removed/ciliate_sags_5kb.fa -o ciliates_sags_5kb_eukrep_euk.fa --prokarya ciliates_sags_5kb_eukrep_pro.fa
#EukRep -i /fs/scratch/PAS0439/Ming/databases/two_stages_classifier/stage2/test/fungi_5kb_concat_test.fasta  -o fungi_test_eukrep_euk.fa --prokarya fungi_test_eukrep_pro.fasta;

DURATION=$(( SECONDS - START ))

echo "Completed in $DURATION seconds."
sacct -j $SLURM_JOB_ID -o JobID,AllocTRES%50,Elapsed,CPUTime,TresUsageInTot,MaxRSS
