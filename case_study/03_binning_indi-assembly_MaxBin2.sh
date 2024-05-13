#!/bin/bash
#SBATCH --job-name=maxbin_%j
#SBATCH --output=maxbin_%j.out
# Walltime Limit: hh:mm:ss 
#SBATCH --time=60:30:00
#SBATCH --nodes=1
#SBATCH --ntasks=12
#SBATCH --mail-type=FAIL
#SBATCH --account=PAS0439

START=$SECONDS

cd  /fs/scratch/PAS0439/Ming/rummeta-370/assembly

module load python/3.6-conda5.2
source activate samtools-1.13
sample_id=$1

coverm contig -t 12 -m trimmed_mean -b ${sample_id}.sorted.bam > ${sample_id}.trimmed_mean.tsv ; 

for f in ${sample_id}.trimmed_mean.tsv;
do echo ${f} >> ${sample_id}_abund_list.txt;
done


source deactivate 
source activate  /users/PAS1855/yan1365/miniconda3/envs/metawrap-env/envs/metawrap

mkdir ../binning/indi-assembly/MaxBin_${sample_id}/

singularity run /users/PAS1117/osu9664/eMicro-Apps/MaxBin2-2.2.6.sif -contig ${sample_id}.final.contigs -abund_list ${sample_id}_abund_list.txt \
-out ../binning/indi-assembly/MaxBin_${sample_id}/MaxBin_${sample_id} -thread 12

DURATION=$(( SECONDS - START ))

echo "Completed in $DURATION seconds."

