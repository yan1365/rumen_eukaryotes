#!/usr/bin/env python3

# Author: Ming Yan, The Ohio State University
import os
import time
import shutil
import utils
import glob
import torch
import argparse
import logging
import subprocess
import multiprocessing
import numpy as np
import pandas as pd
from Bio import SeqIO
from pathlib import Path
from datetime import datetime
from torch.utils.data import DataLoader

description = '''\
GutEuk -- A deep-learning-based two-stage classifier to distinguish contigs/MAGs of prokaryotes, fungi or protozoa origins.

Designed specifically for gut microbiome.

In the first stage, the inputs are classified as either prokaryotes or eukaryotes origin (fungi or protozoa).

In the second stage, the eukaryotic sequences are further classified as either fungi or protozoa.

'''

Usage = '''Usage: GutEuk -i <input_file> -o <output_dir> [options]/ GutEuk -h.
To run GutEuk on a test dataset: GutEuk -i test/test.fa -o ./'''

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


parser.add_argument(
    "--to_fasta", help="Write predicted results to fasta files", action='store_true'
)

args = parser.parse_args()

def main():

    # config variables

    ## setting
    input_fasta = os.path.normpath(args.input)
    fasta_filename = os.path.basename(args.input)
    output_dir = os.path.normpath(args.output_dir)
    min_length = args.min_len
    to_fasta = args.to_fasta

    if min_length > 5000:
        min_length = 5000
    threads = args.threads
    tmp_dir = os.path.normpath(f"{output_dir}/{fasta_filename.split('.')[0]}_GutEuk_tmp")

    ## mise
    start_time = str(datetime.now()).split('.')[0]
    time_spent_start = time.time()

    ## results related
    ### in case of input length greater than 5000, record the origins and results of each 5000 bp fragment
    ### the majority rule is used for the final assignment
    seq_origin = {}  
    seq_assignment = {} 

    if os.path.exists(output_dir):
        pass
    else:
        os.mkdir(output_dir)

    # create a log file
    logging.basicConfig(filename=os.path.join(output_dir, f"{fasta_filename.split('.')[0]}_GutEuk_log.txt"), level=logging.INFO, format='%(message)s')
    logging.info(f"{parser.description}")
    
    # create tmp dir
    try:
        os.mkdir(f'{tmp_dir}')
    except FileExistsError:
        pass
    

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

    # preprocessing
    def preprocessing(input_fasta, tmp_dir, threads, min_length):
        ## split fasta into multiple
        utils.split_fasta_parallel(input_fasta, tmp_dir, threads)
        
        ## fasta to int-encoded
        utils.fasta_int_encoded_parellel(tmp_dir, threads, min_length)

        ## convert int-encoded csv to kmerfre array and onehot-encoded array
        ## the resultant arrays could be used for prediction
        utils.save_npz_parellel(tmp_dir, threads)

    # prediction for inidividual fragment
    def prediction():
        indexs = [f.split("forward_")[1].split(".npz")[0] for f in glob.glob(f"{tmp_dir}/forward*.npz")]
        for index in indexs:
            utils.predict(tmp_dir, index)


    # organize fragment prediction results and generate final output
    def generate_final_output():
        seqorigin = glob.glob(f"{tmp_dir}/seqorigin*.csv")
        stage1_res = glob.glob(f"{tmp_dir}/input_fasta_*_stage1_out.csv")
        stage2_res = glob.glob(f"{tmp_dir}/input_fasta_*_stage2_out.csv")
        if len(seqorigin) == 1:
            seqorigin_df = pd.read_csv(seqorigin[0])
            stage1_df = pd.read_csv(stage1_res[0])
            stage2_df = pd.read_csv(stage2_res[0])

        else:
            seqorigin_df = pd.concat([pd.read_csv(f) for f in seqorigin])
            stage1_df = pd.concat([pd.read_csv(f) for f in stage1_res])
            stage2_df = pd.concat([pd.read_csv(f) for f in stage2_res])

        stage1_res = pd.merge(seqorigin_df, stage1_df, on = "seq")
        stage2_res = pd.merge(seqorigin_df, stage2_df, on = "seq")

        stage1_final_prediction = {}
        for f in set(stage1_res.origin):
            df = stage1_res.query('origin == @f')
            # if the input sequence unfragmented (seq length < 10,000 bp)
            if len(df) == 1:
                prediction = list(df.predict)[0]
                stage1_final_prediction[f] = prediction
            # if the input sequence fragmented, the final prediction for the original sequence determined based on the majority rule
            else:
                prediction = list(df.predict)
                prediction_pro = prediction.count("prokaryotes")
                prediction_euk = prediction.count("eukaryotes") 
                if prediction_pro > prediction_euk:
                    stage1_final_prediction[f] = "prokaryotes"
                elif prediction_pro < prediction_euk:
                    stage1_final_prediction[f] = "eukaryotes"
                else:
                    stage1_final_prediction[f] = "undetermined"

        stage2_final_prediction = {}
        for f in set(stage2_res.origin):
            df = stage2_res.query('origin == @f')
            if len(df) == 1:
                prediction = list(df.predict)[0]
                stage2_final_prediction[f] = prediction
            else:
                prediction = list(df.predict)
                prediction_fungi = prediction.count("fungi")
                prediction_protozoa = prediction.count("protozoa") 
                if prediction_fungi > prediction_protozoa:
                    stage2_final_prediction[f] = "fungi"
                elif prediction_fungi < prediction_protozoa:
                    stage2_final_prediction[f] = "protozoa"
                else:
                    stage2_final_prediction[f] = "undetermined"
        
        stage1_final_prediction_df = pd.DataFrame.from_dict(stage1_final_prediction, orient = "index").rename(columns = {0:"stage1_prediction"})
        stage2_final_prediction_df = pd.DataFrame.from_dict(stage2_final_prediction, orient = "index").rename(columns = {0:"stage2_prediction"})
        final_output = pd.merge(stage1_final_prediction_df, stage2_final_prediction_df, left_index = True, right_index = True, how="left")
        final_output.reset_index(names = "sequence_id", inplace = True)
        final_output.to_csv(f"{output_dir}/{fasta_filename.split('.')[0]}_EukRep_output.csv", index = None)

        return final_output

    
    def write_to_fasta(final_output):
        prokaryotes = list(final_output.query('stage1_prediction == "prokaryotes"').sequence_id)
        eukaryotes = list(final_output.query('stage1_prediction == "eukaryotes"').sequence_id)
        protozoa = list(final_output.query('stage2_prediction == "protozoa"').sequence_id)
        fungi = list(final_output.query('stage2_prediction == "fungi"').sequence_id)

        with open(f"{output_dir}/{fasta_filename.split('.')[0]}_EukRep_prokaryotes.fasta", "w") as prokaryotes_out:
            with open(f"{output_dir}/{fasta_filename.split('.')[0]}_EukRep_eukaryotes.fasta", "w") as eukaryotes_out:
                with open(f"{output_dir}/{fasta_filename.split('.')[0]}_EukRep_protozoa.fasta", "w") as protozoa_out:
                    with open(f"{output_dir}/{fasta_filename.split('.')[0]}_EukRep_fungi.fasta", "w") as fungi_out:
                        records = SeqIO.parse(f"{input_fasta}", "fasta")
                        for record in records:
                            if record.id in prokaryotes:
                                SeqIO.write(record, prokaryotes_out, "fasta") 
                            elif record.id in eukaryotes:
                                SeqIO.write(record, eukaryotes_out, "fasta")
                                if record.id in protozoa:
                                     SeqIO.write(record, protozoa_out, "fasta")
                                elif record.id in fungi:
                                     SeqIO.write(record, fungi_out, "fasta")



    preprocessing_start = time.time()
    preprocessing(input_fasta, tmp_dir, threads, min_length)
    preprocessing_end = time.time()
    logging.info(f"Preprocessing finished in {preprocessing_end - preprocessing_start:.2f} secs")
    

    prediction_start = time.time()
    prediction()
    final_output = generate_final_output()
    if to_fasta:
        write_to_fasta(final_output)
    prediction_end = time.time()
    logging.info(f"Prediction finished in {prediction_end - prediction_start:.2f} secs")


    # clearn up, remove tmp dir
    try:
        shutil.rmtree(f"{tmp_dir}")
    except Exception as e:
        print(f"Unexpected error: {e}")

    end_time = str(datetime.now()).split('.')[0]
    time_spent_end = time.time()

    logging.info(f"Start: {start_time}")
    logging.info(f"End: {end_time}")
    logging.info(f"Time spent: {time_spent_end - time_spent_start:.2f}")


if __name__ == "__main__":
    main()