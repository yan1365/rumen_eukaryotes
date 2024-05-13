#!/bin/bash
#SBATCH --job-name=concoct_%j
#SBATCH --output=concoct_%j.out
# Walltime Limit: hh:mm:ss 
#SBATCH --time=18:30:00
#SBATCH --nodes=1
#SBATCH --ntasks=8
#SBATCH --mail-type=FAIL
#SBATCH --account=PAS0439

START=$SECONDS

cd  /fs/scratch/PAS0439/Ming/rummeta-370/assembly


source activate  samtools-1.13   
sample_id=$1
mkdir ../binning/indi-assembly/Concoct_${sample_id}

samtools index ${sample_id}.sorted.bam


singularity exec /users/PAS1117/osu9664/eMicro-Apps/CONCOCT-1.1.0.sif cut_up_fasta.py ${sample_id}.final.contigs -c 10000 -o 0 --merge_last -b ${sample_id}_contigs_10K.bed > ${sample_id}_contigs_10K.fa

singularity exec /users/PAS1117/osu9664/eMicro-Apps/CONCOCT-1.1.0.sif concoct_coverage_table.py ${sample_id}_contigs_10K.bed ${sample_id}.sorted.bam > ../binning/indi-assembly/Concoct_${sample_id}/${sample_id}_coverage_table.tsv

singularity exec /users/PAS1117/osu9664/eMicro-Apps/CONCOCT-1.1.0.sif concoct --composition_file ${sample_id}_contigs_10K.fa --coverage_file ../binning/indi-assembly/Concoct_${sample_id}/${sample_id}_coverage_table.tsv -b ../binning/indi-assembly/Concoct_${sample_id}/Concoct_${sample_id} -t 8

singularity exec /users/PAS1117/osu9664/eMicro-Apps/CONCOCT-1.1.0.sif merge_cutup_clustering.py ../binning/indi-assembly/Concoct_${sample_id}/Concoct_${sample_id}_clustering_gt1000.csv  > ../binning/indi-assembly/Concoct_${sample_id}/Concoct_${sample_id}_clustering_merged.csv


mkdir ../binning/indi-assembly/Concoct_${sample_id}/fasta_bins
singularity exec /users/PAS1117/osu9664/eMicro-Apps/CONCOCT-1.1.0.sif extract_fasta_bins.py ${sample_id}.final.contigs ../binning/indi-assembly/Concoct_${sample_id}/Concoct_${sample_id}_clustering_merged.csv --output_path ../binning/indi-assembly/Concoct_${sample_id}/fasta_bins

DURATION=$(( SECONDS - START ))

echo "Completed in $DURATION seconds."

