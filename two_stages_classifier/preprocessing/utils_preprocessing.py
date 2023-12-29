import os
import glob
from Bio import SeqIO
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
import sys
import multiprocessing
import shutil
import random
import re
import numpy as np




def chop_sequence(seq, fragment_size = 5000, sliding_window = 5000):
    """Chop a sequence into fragments of equal size"""
    return  [seq[i:i+fragment_size] for i in range(0, len(seq), sliding_window)]
    


def chop_genomes(genome_fasta, output_dir = "./", fragment_size = 5000, sliding_window = 5000, subsample = 1):
    """Chop sequences of genomes into fragments of equal size"""
    genome_name = re.search(r".*/([^/]*).(fasta|fna|fa)", genome_fasta).group(1)
    
    if output_dir[-1] == "/":
        output_dir = output_dir[:-1] # to prevent duplicated /

    output = f"{output_dir}/{genome_name}_{fragment_size}bp.fasta"
    with open(output, "w") as outfile:
        records = SeqIO.parse(genome_fasta, "fasta")
        for record in records:
            id_original = record.id

            fragments =  chop_sequence(str(record.seq).upper().replace("N", ""), fragment_size, sliding_window)
            ## N is removed
            filtered = [str(f).upper() for f in fragments if len(set(f).union(set("ATGC"))) == 4]
            suffix = 1
            for fragment in filtered:
                if len(fragment) == fragment_size:
                    random_choice = random.choices([0, 1], weights=[(1-subsample), subsample], k=1)[0] # randomly choose only 1/10 of the fragments from each genome to reduce the training set size 
                    if random_choice == 1:
                        new_record = SeqRecord(Seq(fragment), id=f"{id_original}_fragment_{suffix}", description="")
                        SeqIO.write(new_record, outfile, "fasta")
                        suffix += 1


def chop_genomes_wrapper(args):
    chop_genomes(*args)


def chop_genomes_parallel(genomes_dir, output_dir = "./", fragment_size = 5000, sliding_window = 5000, subsample = 1, cores=None):
    if cores == None:
        cores = multiprocessing.cpu_count()
    
    fasta_files = glob.glob(f"{genomes_dir}/*.fasta")
    fna_files = glob.glob(f"{genomes_dir}/*.fna")
    fa_files = glob.glob(f"{genomes_dir}/*.fa")
    genomes = fasta_files + fna_files + fa_files

    arg_list = [[genome, output_dir, fragment_size, sliding_window, subsample] for genome in genomes]
   
    with multiprocessing.Pool(processes=cores) as pool:
        pool.map(chop_genomes_wrapper, arg_list)



def sequence_to_5mer_frequency(myseq):
    '''
    input sequence of 5kb chunks, output a dic. the dic.values are the frequence of each kmer
    '''

    nucle = ["A","T","G","C"]
    five_mer = []
    for P1 in nucle:
        for P2 in nucle:
            for P3 in nucle:
                for P4 in nucle:
                    for P5 in nucle:
                        five_mer.append("".join([P1,P2,P3,P4,P5]))
    five_mer_dict = {}
    for f in five_mer:
        five_mer_dict[f] = 0

    #sequence to list of kmer
    kmers = [myseq[i:i+5] for i in range(len(myseq) - (5 -1))]
    kmers_clean = [str(f).upper() for f in kmers if len(set(f).union(set("ATGC"))) == 4] # remove kmer that contains base other than A,T,G or C
    
    kmer_fre_dict = five_mer_dict.copy()
    for kmer in kmers_clean:
        kmer_fre_dict[kmer] += 1

    total_kmers = sum(kmer_fre_dict.values())
    for i in kmer_fre_dict:
        try:
            kmer_fre_dict[i] = kmer_fre_dict[i]/total_kmers
        except ZeroDivisionError:   
            pass 
    return kmer_fre_dict


def genome2kmer(file, output_dir = "./", portion = 1):
    # portion, for random sampling portion of sequence, to downsize
    file_base = re.search(r".*/([^/]*).(fasta|fna|fa)", file).group(1)
    if output_dir[-1] == "/":
        output_dir = output_dir[:-1] # to prevent duplicated /
    output = f"{output_dir}/{file_base}_5mer_frequency.csv"
    with open(output, "w") as handle:
        records = SeqIO.parse(file, "fasta")
        for record in records:
            random_choice = random.choices([0, 1], weights=[(1-portion), portion], k=1)[0] # randomly choose only 1/10 of the fragments from each genome to reduce the training set size 
            if random_choice == 1:
                my_seq = str(record.seq).upper()
                seq_id = str(record.id)
                kmer_fre_dict_tmp =  sequence_to_5mer_frequency(my_seq)
                handle.write(seq_id + "," + ",".join([str(f) for f in list(kmer_fre_dict_tmp.values())]) + "\n")

def genome2kmer_wrapper(args):
    genome2kmer(*args)

def genome2kmer_parallele(genome_dir, output_dir = "./", portion = 1, cores = None):
    # input genome dir of 5kb chunk, output csv of 5mer frequency
    fasta_files = glob.glob(f"{genome_dir}/*.fasta")
    fna_files = glob.glob(f"{genome_dir}/*.fna")
    fa_files = glob.glob(f"{genome_dir}/*.fa")
    genomes = fasta_files + fna_files + fa_files

    
    if cores == None:
        cores = multiprocessing.cpu_count()
           
    
    args_list = [(genome, output_dir, 1) for genome in genomes]

    with multiprocessing.Pool(processes=cores) as pool:
        pool.map(genome2kmer_wrapper, args_list)

def kmerfrequency(x):
    # return dict of 4,5,6-mer frequency
    nucle = [1,2,3,4] # represent A,C,G,T 
    four_mer = []
    for P1 in nucle:
        for P2 in nucle:
            for P3 in nucle:
                for P4 in nucle:
                    four_mer.append("".join([str(P1),str(P2),str(P3),str(P4)]))
    
    five_mer = []
    for P1 in nucle:
        for P2 in nucle:
            for P3 in nucle:
                for P4 in nucle:
                    for P5 in nucle:
                        five_mer.append("".join([str(P1),str(P2),str(P3),str(P4),str(P5)]))
                        
    six_mer = []
    for P1 in nucle:
        for P2 in nucle:
            for P3 in nucle:
                for P4 in nucle:
                    for P5 in nucle:
                        for P6 in nucle:
                            six_mer.append("".join([str(P1),str(P2),str(P3),str(P4),str(P5),str(P6)]))
    
    four_mer_dict = {}
    for f in four_mer:
        four_mer_dict[f] = 0
        
    five_mer_dict = {}
    for f in five_mer:
        five_mer_dict[f] = 0
        
    six_mer_dict = {}
    for f in six_mer:
        six_mer_dict[f] = 0

    kmer_fre_dict = {}

    #sequence to list of kmer
    for kmer in [4, 5, 6]:
        if kmer == 4:
            kmer_dict_tmp = four_mer_dict
        elif kmer == 5:
            kmer_dict_tmp = five_mer_dict
        else:
            kmer_dict_tmp = six_mer_dict
                    
        
        total_kmers = 0
        myseq = "".join([str(f) for f in x])
        kmers = [myseq[i:i+kmer] for i in range(len(myseq) - (kmer -1))]
        kmers_clean = [str(f).upper() for f in kmers if len(set(f).union(set("1234"))) == 4] # remove kmer that contains base other than A,T,G or C

        for kmer in kmers_clean:
            total_kmers += 1
            kmer_dict_tmp[kmer] += 1
        
        for i in kmer_dict_tmp:
            try:
                kmer_dict_tmp[i] = kmer_dict_tmp[i]/total_kmers
            except ZeroDivisionError:   
                pass 

        _ = {**kmer_dict_tmp, **kmer_fre_dict}   
        kmer_fre_dict = _       

    return list(kmer_fre_dict.values())

def nt2int(nt):
    nt = nt.upper()
    if nt == "A":
        return 1
    elif nt == "C":
        return 2
    elif nt == "G":
        return 3
    elif nt == "T":
        return 4
    else:
        return 0

def dna2int(dna):
    return list(map(nt2int, str(dna)))


def one_hot_encode_with_zero_handling(input_array):
    nsamples = input_array.shape[0]
    output_list = []
    for i, _  in enumerate(input_array):
        one_hot_encoded = np.zeros((input_array.shape[1], 4))

        for j, value in enumerate(input_array[i]):
            if value != 0:
                one_hot_encoded[j, value - 1] = 1

        
        output_list.append(one_hot_encoded)

    return np.array(output_list)

def one_hot_reverse_compli(onehot_forward):
    nsamples = onehot_forward.shape[0]
    output_list = []
    for _, i in enumerate(onehot_forward):
        output_list.append(i[::-1,::-1])
    return np.array(output_list)

class dna2onehot_both_dir:
    def __call__(self, sample):
        inputs, targets = sample
        output = []
        forward = utils.one_hot_encode_with_zero_handling(inputs)
        reverse = utils.one_hot_reverse_compli(forward)
        nsamples = forward.shape[0]
        for i in range(nsamples):
            output.append(np.vstack((forward[i], reverse[i])))
        inputs = np.array(output)
        return inputs, targets


