#!/usr/bin/env python3

# Author: Ming Yan, The Ohio State University
import os
import shutil
import utils
import argparse
import logging
import subprocess
from Bio import SeqIO
import time
from datetime import datetime


# define functions args

def dna2onehot(X):
    forward = utils.one_hot_encode_with_zero_handling(X)
    return forward
    
     
def seq2kmerfrequency(X):
    inputs_list = []

    for f in range(len(X)):
        kmerfre = utils.kmerfrequency(X[f,:])
        inputs_list.append(kmerfre)

    inputs_kmerfre = np.array(inputs_list).reshape(len(X), -1)
    return inputs_kmerfre

description = '''\
GutEuk -- A deep-learning-based two-stage classifier to distinguish contigs/MAGs of prokaryotes, fungi or protozoa origin.

Designed specifically for gut microbiome.

In the first stage, the inputs are identified as either prokaryotes or eukaryotes origin (fungi or protozoa).

In the second stage, the eukaryotic sequences are further identified as either fungi or protozoa.

'''

Usage = 'Usage: GutEuk -i <input_file> -o <output_dir> [options]/ GutEuk -h.'

parser = argparse.ArgumentParser(description = description, usage = Usage)

parser.add_argument(
    "-i",
    "--input",
    metavar="input",
    help="The path to a fasta (or gzipped fasta) file.",
    type=str,
    required=True
)

parser.add_argument(
    "-o",
    "--output_dir",
    metavar="output",
    help="A path to output files.",
    default=None,
    required=True
)

parser.add_argument(
    "-m",
    "--min_len",
    help=f"""Minimum length of a sequence. Sequences shorter than min_len are discarded. 
    Default: 3000 bp.""",
    type=int,
    default=3000
)


parser.add_argument(
    "-t", "--threads", help="Number of threads used. The default is 1.", type=int, default=1, nargs=1
)

print(parser.description)
args = parser.parse_args()

def main():

    # config variables

    ## setting
    input_fasta = os.path.normpath(args.input)
    fasta_filename = os.path.basename(args.input)
    output_dir = os.path.normpath(args.output_dir)
    min_length = args.min_len
    threads = args.threads
    tmp_dir = os.path.normpath(f"{output_dir}/tmp")

    ## mise
    start_time = str(datetime.now()).split('.')[0]
    time_spent_start = time.time()

    ## results related
    ### in case of input length greater than 5000, record the origins and results of each 5000 bp fragment
    ### the majority rule is used for the final assignment
    seq_origin = {}  
    seq_assignment = {} 
    
    # create tmp dir
    try:
        os.mkdir(f'{tmp_dir}')
    except FileExistsError:
        pass

    # create a log file
    logging.basicConfig(filename=os.path.join(output_dir, "log.txt"), level=logging.INFO, format='%(message)s')
    logging.info(f"{parser.description}")
    

    # unzip if zipped
    if input_fasta.endswith(".gz"):
        copy = f"cp {input_fasta} {tmp_dir}"
        gunzip = f"gunzip {tmp_dir}/{fasta_filename}"
        subprocess.run(copy, shell=True, check=True, text=True)
        subprocess.run(gunzip, shell=True, check=True, text=True)
        fasta_filename = fasta_filename.strip(".gz")
        input_fasta = os.path.join(tmp_dir, fasta_filename)
    else:
        copy = f"cp {input_fasta} {tmp_dir}"


    # data transformation
    with open(f"{tmp_dir}/input_fasta.csv", "w") as handle:
        records = SeqIO.parse(input_fasta, "fasta")
        for record in records:
            seqid = record.id
            seq = record.seq
            intencoded = utils.dna2int(seq)
            if len(intencoded) < min_length:
                continue
            elif len(intencoded) <= 5000:
                # zero-padding on the right to 5000 bp when seq length < 5000
                for f in range(5000 - len(intencoded)):
                    intencoded.append(0)
                seq_origin[str(seqid)] = str(seqid)
                handle.write(f"{seqid}," +  ",".join([str(f) for f in intencoded]) + "\n")
            
            else:
                # if seq length > 5000, split it into fragments of 5000 bp and give prediction to each fragment 
                fragments =  len(intencoded) // 5000  
                if fragments == 1:
                    intencoded = intencoded[:5000]
                    seq_origin[str(seqid)] = str(seqid)
                    handle.write(f"{seqid}," +  ",".join([str(f) for f in intencoded]) + "\n")
                
                else:
                    for f in range(fragments):
                        intencoded_fragment = intencoded[5000*f:5000*(f+1)]
                        seqid_new = f"{str(seqid)}_{str(f+1)}"
                        seq_origin[str(seqid_new)] = str(seqid)
                        handle.write(f"{seqid_new}," +  ",".join([str(f) for f in intencoded_fragment]) + "\n")












    # # clearn up, remove tmp dir
    # try:
    #     shutil.rmtree(f"{tmp_dir}")
    #     print(f"Tmp directory '{tmp_dir}' and its contents removed successfully.")
    # except Exception as e:
    #     print(f"Error removing directory '{tmp_dir}': {e}")

    end_time = str(datetime.now()).split('.')[0]
    time_spent_end = time.time()

    logging.info(f"Start: {start_time}")
    logging.info(f"End: {end_time}")
    logging.info(f"Time spent: {time_spent_end - time_spent_start}")



if __name__ == "__main__":
    main()









