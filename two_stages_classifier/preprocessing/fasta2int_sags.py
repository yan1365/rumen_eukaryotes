import os
import numpy as np
import pandas as pd
from Bio import SeqIO
import sys
import numpy as np
import utils_preprocessing as utils


## convert dataset of 5kb chunk DNA sequence to int sequence
os.chdir("/fs/scratch/PAS0439/Ming/databases/gut_eukaryotes_classifier/")
with open("train/sags_train_5000bp.csv", "w") as handle:
    records = SeqIO.parse("train/sags_train_5000bp.fasta", "fasta")
    for record in records:
        seqid = record.id
        seq = record.seq
        intencoded = utils.dna2int(seq)
        handle.write(f"{seqid}," +  ",".join([str(f) for f in intencoded]) + f",1\n")

with open("val/sags_val_5000bp.csv", "w") as handle:
    records = SeqIO.parse("val/sags_val_5000bp.fasta", "fasta")
    for record in records:
        seqid = record.id
        seq = record.seq
        intencoded = utils.dna2int(seq)
        handle.write(f"{seqid}," +  ",".join([str(f) for f in intencoded]) + f",1\n")

with open("test/sags_test_5000bp.csv", "w") as handle:
    records = SeqIO.parse("test/sags_test_5000bp.fasta", "fasta")
    for record in records:
        seqid = record.id
        seq = record.seq
        intencoded = utils.dna2int(seq)
        handle.write(f"{seqid}," +  ",".join([str(f) for f in intencoded]) + f",0\n")

