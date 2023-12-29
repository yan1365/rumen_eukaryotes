import os
import numpy as np
import pandas as pd
from Bio import SeqIO
import sys
import numpy as np
import utils_preprocessing as utils


## convert dataset of 5kb chunk DNA sequence to int sequence
os.chdir("/fs/scratch/PAS0439/Ming/databases/gut_eukaryotes_classifier/train")
with open("protozoa_5kb_concat.csv", "w") as handle:
    records = SeqIO.parse("protozoa_5kb_concat.fasta", "fasta")
    for record in records:
        seqid = record.id
        seq = record.seq
        intencoded = utils.dna2int(seq)
        handle.write(f"{seqid}," +  ",".join([str(f) for f in intencoded]) + f",1\n")

with open("fungi_5kb_concat.csv", "w") as handle:
    records = SeqIO.parse("fungi_5kb_concat.fasta", "fasta")
    for record in records:
        seqid = record.id
        seq = record.seq
        intencoded = utils.dna2int(seq)
        handle.write(f"{seqid}," +  ",".join([str(f) for f in intencoded]) + f",1\n")

with open("proka_5kb_concat.csv", "w") as handle:
    records = SeqIO.parse("proka_5kb_concat.fasta", "fasta")
    for record in records:
        seqid = record.id
        seq = record.seq
        intencoded = utils.dna2int(seq)
        handle.write(f"{seqid}," +  ",".join([str(f) for f in intencoded]) + f",0\n")

