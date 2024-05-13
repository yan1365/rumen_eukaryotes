#!/bin/bash
#SBATCH --job-name=db_%j
#SBATCH --output=db_%j.out
# Walltime Limit: hh:mm:ss 
#SBATCH --time=20:30:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mail-type=ALL
#SBATCH --account=PAS0439

START=$SECONDS
cd /fs/ess/PAS0439/MING/databases/uniclust50
wget https://wwwuser.gwdg.de/~compbiol/uniclust/2023_02/UniRef30_2023_02_hhsuite.tar.gz

DURATION=$(( SECONDS - START ))

echo "Completed in $DURATION seconds."
