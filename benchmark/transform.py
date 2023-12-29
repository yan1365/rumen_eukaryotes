import os
import numpy as np
import pandas as pd
from Bio import SeqIO
import sys
sys.path.append('/users/PAS1855/yan1365/rumen_eukaryotes/eukaryotes_classifier/model/preprocessing')
import numpy as np
import utils_preprocessing as utils


## convert dataset of 5kb chunk DNA sequence to int sequence
os.chdir("/fs/ess/PAS0439/MING/cilates_fungi_classifier/outputs")
with open("ciliates_sags_5kb.csv", "w") as handle:
    records = SeqIO.parse("ciliates_sags_5kb.fa", "fasta")
    for record in records:
        seqid = record.id
        seq = record.seq
        intencoded = utils.dna2int(seq)
        handle.write(f"{seqid}," +  ",".join([str(f) for f in intencoded]) + f",2\n")

with open("ruminant_fungi_5kb.csv", "w") as handle:
    records = SeqIO.parse("ruminant_fungi_5kb.fa", "fasta")
    for record in records:
        seqid = record.id
        seq = record.seq
        intencoded = utils.dna2int(seq)
        handle.write(f"{seqid}," +  ",".join([str(f) for f in intencoded]) + f",1\n")



