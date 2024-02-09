#!/bin/bash
#SBATCH --job-name=benchmark_GutEuk_fun_%j
#SBATCH --output=benchmark_GutEuk_fun_%j.out
# Walltime Limit: hh:mm:ss 
#SBATCH --time=20:20:00
#SBATCH --nodes=1
#SBATCH --ntasks=20 
#SBATCH --mail-type=ALL
#SBATCH --account=PAS0439

START=$SECONDS

module load python/3.6-conda5.2
source activate  torch 


#python ../GutEuk/GutEuk.py -i /fs/scratch/PAS0439/Ming/databases/gut_eukaryotes_classifier/test/proka.fasta -o /fs/scratch/PAS0439/Ming/GutEuk_benchmark/ -t 20 

#python ../GutEuk/GutEuk.py -i /fs/scratch/PAS0439/Ming/databases/gut_eukaryotes_classifier/test/protozoa.fasta -o /fs/scratch/PAS0439/Ming/GutEuk_benchmark/ -t 20 

#python ../GutEuk/GutEuk.py -i /fs/scratch/PAS0439/Ming/databases/gut_eukaryotes_classifier/test/fungi.fasta -o /fs/scratch/PAS0439/Ming/GutEuk_benchmark/ -t 20 

python ../GutEuk/GutEuk.py -i /fs/scratch/PAS0439/Ming/databases/gut_eukaryotes_classifier/test/sags_test_chopped.fasta -o /fs/scratch/PAS0439/Ming/GutEuk_benchmark/ -t 20 

DURATION=$(( SECONDS - START ))

echo "Completed in $DURATION seconds."
sacct -j $SLURM_JOB_ID -o JobID,AllocTRES%50,Elapsed,CPUTime,TresUsageInTot,MaxRSS