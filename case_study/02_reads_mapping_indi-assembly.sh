#!/bin/bash
#SBATCH --job-name=mapping_%j
#SBATCH --output=mapping_%j.out
# Walltime Limit: hh:mm:ss 
#SBATCH --time=18:30:00
#SBATCH --nodes=1
#SBATCH --ntasks=8
#SBATCH --mail-type=FAIL
#SBATCH --account=PAS0439

START=$SECONDS

cd /fs/scratch/PAS0439/Ming/rummeta-370/assembly

module load python/3.6-conda5.2
source activate samtools-1.13
sample_id=$1
# indexing
bowtie2-build -f ${sample_id}.final.contigs ${sample_id}_indi-assembly_bowtie2 -p 8

# mapping
bowtie2 -q --phred33 --end-to-end -p 40 -I 0 -X 2000 --no-unal -x ${sample_id}_indi-assembly_bowtie2 -1 ../clean_reads/${sample_id}_1.fastq.gz -2 ../clean_reads/${sample_id}_2.fastq.gz -S ${sample_id}.sam -p 8

# sam to bam
samtools view -Sb ${sample_id}.sam > ${sample_id}.bam -@ 8

# sort bam
samtools sort -o ${sample_id}.sorted.bam  ${sample_id}.bam -@ 8

DURATION=$(( SECONDS - START ))

echo "Completed in $DURATION seconds."

