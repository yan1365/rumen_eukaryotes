#!/bin/bash
#SBATCH --job-name=ncbi_%j
#SBATCH --output=ncbi_%j.out
# Walltime Limit: hh:mm:ss 
#SBATCH --time=20:30:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mail-type=ALL
#SBATCH --account=PAS0439

START=$SECONDS
# downloaded 10-2-2023

source activate ncbi_datasets

cd /fs/ess/PAS0439/MING/databases/protozoa_genome_genbank

for f in $(cat accession.txt);
do datasets download genome accession $f --include genome;
mv ncbi_dataset.zip ${f}.zip;
done



DURATION=$(( SECONDS - START ))

echo "Completed in $DURATION seconds."
sacct -j $SLURM_JOB_ID -o JobID,AllocTRES%50,Elapsed,CPUTime,TresUsageInTot,MaxRSS