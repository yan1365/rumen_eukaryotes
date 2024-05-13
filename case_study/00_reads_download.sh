#!/bin/bash
#SBATCH --job-name=fastp_%j
#SBATCH --output=fastp_%j.out
# Walltime Limit: hh:mm:ss 
#SBATCH --time=20:30:00
#SBATCH --nodes=1
#SBATCH --ntasks=3
#SBATCH --mail-type=ALL
#SBATCH --account=PAS0439

START=$SECONDS
module load python/3.6-conda5.2
source activate sra_tools-2.11

inputs=$1
for f in $(cat $inputs);
do source activate sra_tools-2.11;
cd /fs/scratch/PAS0439/Ming/databases/metatranscriptomics/pair-ended;
fasterq-dump $f;
conda deactivate;
source activate fastp;
fastp -i ${f}_1.fastq -I ${f}_2.fastq -o ${f}_1.fq.gz -O ${f}_2.fq.gz
rm ${f}_1.fastq ${f}_2.fastq;
conda deactivate;
done

DURATION=$(( SECONDS - START ))

echo "Completed in $DURATION seconds."
