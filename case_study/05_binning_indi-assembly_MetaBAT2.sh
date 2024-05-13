#!/bin/bash
#SBATCH --job-name=matebat_%j
#SBATCH --output=matebat_%j.out
# Walltime Limit: hh:mm:ss 
#SBATCH --time=12:30:00
#SBATCH --nodes=1
#SBATCH --ntasks=8
#SBATCH --mail-type=FAIL
#SBATCH --account=PAS0439

START=$SECONDS

cd /fs/scratch/PAS0439/Ming/rummeta-370/assembly

module load python/3.6-conda5.2
source activate  /users/PAS1855/yan1365/miniconda3/envs/metawrap-env/envs/metawrap
sample_id=$1
mkdir /fs/scratch/PAS0439/Ming/rummeta-370/binning/indi-assembly/MetaBat_${sample_id}/

singularity exec /users/PAS1117/osu9664/eMicro-Apps/MetaBAT2-2.14.sif metabat2 -i ${sample_id}.final.contigs ${sample_id}_P1.sorted.bam ${sample_id}_P2.sorted.bam -o /fs/scratch/PAS0439/Ming/rummeta-370/binning/indi-assembly/MetaBat_${sample_id}/MetaBat_${sample_id}

DURATION=$(( SECONDS - START ))

echo "Completed in $DURATION seconds."

