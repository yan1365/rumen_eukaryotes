#!/bin/bash
#SBATCH --job-name=ncbi_%j
#SBATCH --output=ncbi_%j.out
# Walltime Limit: hh:mm:ss 
#SBATCH --time=1:30:00
#SBATCH --nodes=1
#SBATCH --ntasks=20
#SBATCH --mail-type=ALL
#SBATCH --account=PAS0439

START=$SECONDS
# downloaded 10-2-2023

module load python/3.6-conda5.2
source activate  ncbi_genome_download

ncbi-genome-download  protozoa  -p 20 --format fasta

DURATION=$(( SECONDS - START ))

echo "Completed in $DURATION seconds."
sacct -j $SLURM_JOB_ID -o JobID,AllocTRES%50,Elapsed,CPUTime,TresUsageInTot,MaxRSS