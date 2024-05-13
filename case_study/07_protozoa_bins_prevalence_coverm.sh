#!/bin/bash
#SBATCH --job-name=coverm_%j
#SBATCH --output=coverm_%j.out
# Walltime Limit: hh:mm:ss 
#SBATCH --time=80:30:00
#SBATCH --nodes=1
#SBATCH --ntasks=20
#SBATCH --mail-type=END
#SBATCH --account=PAS0439

START=$SECONDS
source activate coverm
file=$1
for f in $(cat $file);
# do coverm genome --coupled /fs/scratch/PAS0439/Ming/rumen975/cleanreads/${f}  /fs/scratch/PAS0439/Ming/rumen975/cleanreads/${f%_1.fq.gz}_2.fq.gz \
# --bam-file-cache-directory /fs/scratch/PAS0439/Ming/tmp \
# -d /fs/ess/PAS0439/MING/cilates_fungi_classifier/results/protozoa_bins/medium_quality_bins --min-read-percent-identity 0.95 --min-read-aligned-percent 0.75  -m covered_fraction \
# --discard-unmapped -t 20 > /fs/ess/PAS0439/MING/cilates_fungi_classifier/results/protozoa_bins/medium_quality_bins/coverm_res/${f%_1.fq.gz}.txt;
# done

# do coverm genome --coupled /fs/scratch/PAS0439/Ming/rumen975/cleanreads/${f}  /fs/scratch/PAS0439/Ming/rumen975/cleanreads/${f%_1.fq.gz}_2.fq.gz \
# --bam-file-cache-directory /fs/scratch/PAS0439/Ming/tmp \
# -d /fs/ess/PAS0439/MING/databases/ciliates_SAGs/high_quality --min-read-percent-identity 0.95 --min-read-aligned-percent 0.75  -m covered_fraction \
# --discard-unmapped -t 20 > /fs/ess/PAS0439/MING/cilates_fungi_classifier/results/protozoa_bins/medium_quality_bins/coverm_res/SAGs/${f%_1.fq.gz}.txt -x fa;
# done

# do coverm genome --coupled /fs/scratch/PAS0439/Ming/rummeta-370/clean_reads/${f}  /fs/scratch/PAS0439/Ming/rummeta-370/clean_reads/${f%_1.fastq.gz}_2.fastq.gz \
# --bam-file-cache-directory /fs/scratch/PAS0439/Ming/tmp \
# -d /fs/ess/PAS0439/MING/cilates_fungi_classifier/results/protozoa_bins/medium_quality_bins --min-read-percent-identity 0.95 --min-read-aligned-percent 0.75  -m covered_fraction \
# --discard-unmapped -t 20 > /fs/ess/PAS0439/MING/cilates_fungi_classifier/results/protozoa_bins/medium_quality_bins/coverm_res/${f%_1.fastq.gz}.txt;
# done

do coverm genome --coupled /fs/scratch/PAS0439/Ming/rummeta-370/clean_reads/${f}  /fs/scratch/PAS0439/Ming/rummeta-370/clean_reads/${f%_1.fastq.gz}_2.fastq.gz \
--bam-file-cache-directory /fs/scratch/PAS0439/Ming/tmp \
-d /fs/ess/PAS0439/MING/databases/ciliates_SAGs/high_quality --min-read-percent-identity 0.95 --min-read-aligned-percent 0.75  -m covered_fraction \
--discard-unmapped -t 20 > /fs/ess/PAS0439/MING/cilates_fungi_classifier/results/protozoa_bins/medium_quality_bins/coverm_res/SAGs/${f%_1.fastq.gz}.txt -x fa;
done

DURATION=$(( SECONDS - START ))

echo "Completed in $DURATION seconds."
