import os
import glob
from Bio import SeqIO
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord

os.chdir("/fs/ess/PAS0439/MING/databases/ciliates_SAGs/telomere_capped/telomere_removed")

def chop_sequence(seq, fragment_size):
    """Chop a sequence into fragments of equal size."""
    return [seq[i:i+fragment_size] for i in range(0, len(seq), fragment_size)]

capped = glob.glob("*.genomic.fa")

for file in capped:
    file_name = file.split("/")[-1].split('.fa')[0]
    with open(f"./{file_name}_5kb.fa", "w") as outfile:
        records = SeqIO.parse(file, "fasta")
        for record in records:
            id_original = record.id

            fragments =  chop_sequence(record.seq, 5000)
            suffix = 1
            for fragment in fragments:

                if len(fragment) == 5000:

                    new_record = SeqRecord(fragment, id=f"{id_original}_fragment_{suffix}", description="")
                    SeqIO.write(new_record, outfile, "fasta")
                    suffix += 1

os.chdir("/fs/ess/PAS0439/MING/databases/ruminant_fungi")

fungi = glob.glob("*.fasta")

for file in fungi:
    file_name = file.split("/")[-1].split('.fasta')[0]
    with open(f"./{file_name}_5kb.fa", "w") as outfile:
        records = SeqIO.parse(file, "fasta")
        for record in records:
            id_original = record.id

            fragments =  chop_sequence(record.seq, 5000)
            suffix = 1
            for fragment in fragments:

                if len(fragment) == 5000:

                    new_record = SeqRecord(fragment, id=f"{id_original}_fragment_{suffix}", description="")
                    SeqIO.write(new_record, outfile, "fasta")
                    suffix += 1
