#!/usr/bin/env python3

# Author: Ming Yan, The Ohio State University
import os
import time
import shutil
import utils
import glob
import argparse
import logging
import subprocess
import multiprocessing
from Bio import SeqIO
from datetime import datetime

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
    "-t", "--threads", help="Number of threads used. The default is 1.", type=int, default=1
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
    

    # unzip {input_fasta} if zipped
    if input_fasta.endswith(".gz"):
        copy = f"cp {input_fasta} {tmp_dir}"
        gunzip = f"gunzip {tmp_dir}/{fasta_filename}"
        subprocess.run(copy, shell=True, check=True, text=True)
        subprocess.run(gunzip, shell=True, check=True, text=True)
        fasta_filename = fasta_filename.strip(".gz")
        input_fasta = os.path.join(tmp_dir, fasta_filename)
    else:
        copy = f"cp {input_fasta} {tmp_dir}"

    
    def preprocessing(input_fasta, tmp_dir, threads, min_length):
        ## split fasta into multiple
        utils.split_fasta_parallel(input_fasta, tmp_dir, threads)
        
        ## fasta to int-encoded
        utils.fasta_int_encoded_parellel(tmp_dir, threads, min_length)

        ## convert int-encoded csv to kmerfre array and onehot-encoded array
        ## the resultant arrays could be used for prediction
        utils.save_npz_parellel(tmp_dir, threads)

    preprocessing(input_fasta, tmp_dir, threads, min_length)
        
        













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









