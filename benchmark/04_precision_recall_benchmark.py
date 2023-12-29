import os
import pandas as pd
from Bio import SeqIO
import sys

sys.path.append('../two_stages_classifier/training/')
import utils
from utils import precision_recall

os.chdir("/fs/ess/PAS0439/MING/cilates_fungi_classifier/outputs/downstream_analysis/benchmark/")

seqorigin = pd.read_csv("/fs/ess/PAS0439/MING/cilates_fungi_classifier/testset_seq_origin.csv")

testing_pro = list(pd.read_csv("../../../dataset_proka_benchmark.csv").id)
testing_fungi = list(pd.read_csv("../../../dataset_fungi_benchmark.csv").files)
testing_protozoa = list(pd.read_csv("../../../dataset_protozoa_benchmark.csv").id)
testing_sags = list(seqorigin.query('category == "SAGs"').genome)
testing_genomes = testing_pro + testing_fungi + testing_protozoa + testing_sags
testing_set = seqorigin[seqorigin.genome.isin(testing_genomes)]

tiara_pro = list(pd.read_csv('pro_test_tiara.txt', sep = "\t").sequence_id)
tiara_euk = list(pd.read_csv('euk_test_tiara.txt', sep = "\t").sequence_id)

eukrep_pro = []
eukrep_euk = []
records = SeqIO.parse("test_eukrep_euk.fa", "fasta") 
for record in records:
    eukrep_euk.append(str(record.id))
    
records = SeqIO.parse("test_eukrep_pro.fa", "fasta") 
for record in records:
    eukrep_pro.append(str(record.id))


tiara_out = []
eukrep_out = []
Y = []
for index, row in testing_set.iterrows():
    cate = row["category"]
    if cate == "prokaryotes":
        Y.append(0)
    else:
        Y.append(1)
        
    seq = row['seq']
    if seq in tiara_pro:
        tiara_out.append(0)
    elif seq in tiara_euk:
        tiara_out.append(1)
        
    if seq in eukrep_pro:
        eukrep_out.append(0)
    elif seq in eukrep_euk:
        eukrep_out.append(1)

    
    
testing_set["tiara_out"] = tiara_out
testing_set["eukrep_out"] = eukrep_out
testing_set["Y"] = Y
testing_set.to_csv("/fs/ess/PAS0439/MING/cilates_fungi_classifier/testset_res_tiara_eukrep.csv", index = None)
