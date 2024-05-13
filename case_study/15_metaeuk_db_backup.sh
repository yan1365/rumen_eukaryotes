#!/bin/bash
#SBATCH --job-name=metaeuk_%j
#SBATCH --output=metaeuk_%j.out
# Walltime Limit: hh:mm:ss 
#SBATCH --time=40:30:00
#SBATCH --nodes=1
#SBATCH --ntasks=40
#SBATCH --mail-type=ALL
#SBATCH --account=PAS0439

START=$SECONDS
source activate /fs/ess/PAS0439/MING/conda/metaeuk
# downloaded 2/28/2024
cd  /fs/ess/PAS0439/MING/databases/
mmseqs createdb /fs/ess/PAS0439/MING/cilates_fungi_classifier/results/eukaryotes_contigs.fa  /fs/ess/PAS0439/MING/cilates_fungi_classifier/results/eukaryotes_contigs.DB

mmseqs databases UniProtKB/Swiss-Prot  UniProtKB_Swiss-Prot/Swiss-prot  /fs/scratch/PAS0439/Ming/tmp
mmseqs createdb rumen_eukaryotes_protein/rumen_eukaryotes_protein.fasta  rumen_eukaryotes_protein/rumen_eukaryotes_protein.DB
mmseqs linclust rumen_eukaryotes_protein/rumen_eukaryotes_protein.DB  rumen_eukaryotes_protein/rumen_eukaryotes_protein.DB_clu /fs/scratch/PAS0439/Ming/tmp
mmseqs clusterupdate  rumen_eukaryotes_protein/rumen_eukaryotes_protein.DB UniProtKB_Swiss-Prot/Swiss-prot  rumen_eukaryotes_protein/rumen_eukaryotes_protein.DB_clu  mmseqs_db_unipro_swiss_prot_plus_rumen_eukaryotes_protein/unipro_swiss_prot_plus_rumen_eukaryotes_protein.DB mmseqs_db_unipro_swiss_prot_plus_rumen_eukaryotes_protein/unipro_swiss_prot_plus_rumen_eukaryotes_protein.DB_clu /fs/scratch/PAS0439/Ming/tmp
DURATION=$(( SECONDS - START ))

echo "Completed in $DURATION seconds."
