#!/bin/bash
#SBATCH --job-name=Plass_%j
#SBATCH --output=Plass_%j.out
# Walltime Limit: hh:mm:ss 
#SBATCH --time=10:30:00
#SBATCH --nodes=1
#SBATCH --ntasks=40
#SBATCH --mail-type=ALL
#SBATCH --account=PAS0439

START=$SECONDS
source activate plass
f=$1
echo $f
cd /fs/scratch/PAS0439/Ming/databases/metatranscriptomics/polyA-enriched;
mkdir ../plass_out/${f}
plass assemble ${f}_1.fq.gz ${f}_2.fq.gz ../plass_out/${f} tmp;
mv ../plass_out/${f}/assembly.fasta ../plass_out/${f}.fasta;

DURATION=$(( SECONDS - START ))

echo "Completed in $DURATION seconds."
