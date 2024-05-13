#!/bin/bash
#SBATCH --job-name=metaeuk_%j
#SBATCH --output=metaeuk_%j.out
# Walltime Limit: hh:mm:ss 
#SBATCH --time=6:30:00
#SBATCH --nodes=1
#SBATCH --ntasks=40
#SBATCH --mail-type=ALL
#SBATCH --account=PAS0439

START=$SECONDS
source activate /fs/ess/PAS0439/MING/conda/metaeuk

cd  /fs/ess/PAS0439/MING/databases/

# create db
mmseqs createdb rumen_eukaryotes_protein/rumen_eukaryotes_protein.fasta  rumen_eukaryotes_protein/rumen_eukaryotes_protein.DB # for rumen
mmseqs databases UniProtKB/Swiss-Prot  UniProtKB_Swiss-Prot/Swiss-prot  /fs/scratch/PAS0439/Ming/tmp # for hindgut

# cluster query db 90% sequence identity, over 80% length with cov-mode 0
mmseqs linclust rumen_eukaryotes_protein/rumen_eukaryotes_protein.DB  rumen_eukaryotes_protein/rumen_eukaryotes_protein.DB_clu /fs/scratch/PAS0439/Ming/tmp

# extract the representative sequences from the clustering result and create db
mmseqs createsubdb rumen_eukaryotes_protein/rumen_eukaryotes_protein.DB_clu  rumen_eukaryotes_protein/rumen_eukaryotes_protein.DB  rumen_eukaryotes_protein/rumen_eukaryotes_protein.DB_clu_repre
mmseqs convert2fasta rumen_eukaryotes_protein/rumen_eukaryotes_protein.DB_clu_repre  rumen_eukaryotes_protein/rumen_eukaryotes_protein.DB_clu_repre.fasta
mmseqs createdb rumen_eukaryotes_protein/rumen_eukaryotes_protein.DB_clu_repre.fasta  rumen_eukaryotes_protein/rumen_eukaryotes_protein.DB_clu90

DURATION=$(( SECONDS - START ))

echo "Completed in $DURATION seconds."
