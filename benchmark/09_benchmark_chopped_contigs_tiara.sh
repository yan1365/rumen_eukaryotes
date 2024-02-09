#!/bin/bash
#SBATCH --job-name=tiara_%j
#SBATCH --output=tiara_%j.out
# Walltime Limit: hh:mm:ss 
#SBATCH --time=00:30:00
#SBATCH --nodes=1                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                              
#SBATCH --ntasks-per-node=20 
#SBATCH --account=PAS0439
#SBATCH --mail-type=ALL

module load python/3.6-conda5.2 
source activate /fs/scratch/PAS0439/Ming/conda/tiara


#tiara -t 20 -i /fs/scratch/PAS0439/Ming/databases/gut_eukaryotes_classifier/test/proka.fasta  -o /fs/scratch/PAS0439/Ming/GutEuk_benchmark/proka_tiara.txt
#tiara -t 20 -i /fs/scratch/PAS0439/Ming/databases/gut_eukaryotes_classifier/test/fungi.fasta  -o /fs/scratch/PAS0439/Ming/GutEuk_benchmark/fungi_tiara.txt
#tiara -t 20 -i /fs/scratch/PAS0439/Ming/databases/gut_eukaryotes_classifier/test/protozoa.fasta  -o /fs/scratch/PAS0439/Ming/GutEuk_benchmark/protozoa_tiara.txt
tiara -t 20 -i /fs/scratch/PAS0439/Ming/databases/gut_eukaryotes_classifier/test/sags_test_chopped.fasta  -o /fs/scratch/PAS0439/Ming/GutEuk_benchmark/sags_tiara.txt


DURATION=$(( SECONDS - START ))

echo "Completed in $DURATION seconds."
sacct -j $SLURM_JOB_ID -o JobID,AllocTRES%50,Elapsed,CPUTime,TresUsageInTot,MaxRSS
