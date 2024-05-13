#!/bin/bash
#SBATCH --job-name=mmseqs2_%j
#SBATCH --output=mmseqs2_%j.out
# Walltime Limit: hh:mm:ss 
#SBATCH --time=20:30:00
#SBATCH --nodes=1
#SBATCH --ntasks=40
#SBATCH --mail-type=ALL
#SBATCH --account=PAS0439

START=$SECONDS
source activate /fs/ess/PAS0439/MING/conda/metaeuk
input=$1
cd /fs/ess/PAS0439/MING/cilates_fungi_classifier/results/metaeuk_out

# create db
mmseqs createdb ${input}  databases/${input%.faa}.DB

# cluster query db 50% sequence identity, over 80% length with cov-mode 0
mmseqs linclust databases/${input%.faa}.DB  databases/${input%.faa}.DB_clu --min-seq-id 0.5 /fs/scratch/PAS0439/Ming/tmp 

# extract the representative sequences from the clustering result and create db
mmseqs createsubdb databases/${input%.faa}.DB_clu  databases/${input%.faa}.DB  databases/${input%.faa}.DB_clu_repre
mmseqs convert2fasta databases/${input%.faa}.DB_clu_repre  databases/${input%.faa}_clu50.faa

DURATION=$(( SECONDS - START ))

echo "Completed in $DURATION seconds."
