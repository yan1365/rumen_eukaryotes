#!/bin/bash
#SBATCH --job-name=fastp_%j
#SBATCH --output=fastp_%j.out
# Walltime Limit: hh:mm:ss 
#SBATCH --time=16:30:00
#SBATCH --nodes=1
#SBATCH --ntasks=4
#SBATCH --mail-type=ALL
#SBATCH --account=PAS0439

START=$SECONDS
module load python/3.6-conda5.2
source activate fastp

inputs=$1
for f in $(cat $inputs);
do r1=$f;
r2=${f%_1.fastq.gz}_2.fastq.gz;
fastp -i /fs/scratch/PAS0439/Ming/rummeta-370/raw_reads/${r1} -I /fs/scratch/PAS0439/Ming/rummeta-370/raw_reads/${r2} -o /fs/scratch/PAS0439/Ming/rummeta-370/clean_reads/${r1} -O /fs/scratch/PAS0439/Ming/rummeta-370/clean_reads/${r2};
done

DURATION=$(( SECONDS - START ))

echo "Completed in $DURATION seconds."
