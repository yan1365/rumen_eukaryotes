#!/usr/bin/env python3

# Author: Ming Yan, The Ohio State University
import os
import re
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
    help='''The path to the contigs file (FASTA or gzipped FASTA).
    If -b/--bin is provided, the input should be the dir containing individual bins (MAGs).''',
    type=str,
    required=True
)


parser.add_argument(
    "-b",
    "--bin",
    help="If provided, treat individual FASTA files as bins instead of contigs",
    required=False,
    action='store_true'
)

parser.add_argument(
    "-s1",
    "--stage1_confidence",
    metavar="stage1 confidence level",
    help='''Confidence level for stage1 classification: 
    e.g. -s1 0.6: only give predictions when 60 percents of the contigs/bins are classified as the same category.
    Default: 0.5.
    ''',
    required=False,
    type=float,
    default=0.5
)


parser.add_argument(
    "-s2",
    "--stage2_confidence",
    metavar="stage2 confidence level",
    help='''Confidence level for stage2 classification: 
    e.g. -s2 0.6: only give predictions when 60 percents of the contigs/bins are classified as the same category.
    Default: 0.5.
    ''',
    required=False,
    type=float,
    default=0.5
)


parser.add_argument(
    "-o",
    "--output_dir",
    metavar="output",
    help='''A path to output files.
    If -b/--bin is provided, the output is a csv file containing prediction results for the input bins (MAGs)''',
    default=None,
    required=True
)

parser.add_argument(
    "-m",
    "--min_len",
    help=f"""Minimum length of a sequence. Sequences shorter than min_len are discarded. 
    Default: 5000 bp.""",
    type=int,
    default=5000
)


parser.add_argument(
    "-t", "--threads", help="Number of threads used. Default: 1.", type=int, default=1
)


parser.add_argument(
    "--to_fasta", help="Write predicted results to fasta files (for contigs only)", action='store_true'
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
    input_bin = args.bin
    
    if input_bin:
        tmp_dir = f"{output_dir}/tmp"

    else:
        fasta_filename_trailing_removed = re.search(r"(.*).(fa|fasta|fna)$", fasta_filename).group(1)
        tmp_dir = os.path.normpath(f"{output_dir}/{fasta_filename_trailing_removed}_GutEuk_tmp")


    threads = args.threads
    s1 = args.stage1_confidence
    s2 = args.stage2_confidence
   

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

    # create a log file, overwrite if existed
    if input_bin:
        logfile = f"{fasta_filename}_GutEuk_log.txt"
    else:
        logfile = f"{output_dir}/{fasta_filename_trailing_removed}_GutEuk_log.txt"
    if os.path.exists(logfile):
        os.remove(logfile)
    logging.basicConfig(filename=os.path.join(logfile), level=logging.INFO, format='%(message)s')
    logging.info(f"{parser.description}")
    
    # create tmp dir
    try:
        os.mkdir(f'{tmp_dir}')
    except FileExistsError:
        pass
    

    # preprocessing 
    def preprocessing(input_fasta, tmp_dir, threads, min_length):
        ## split fasta into multiple
        utils.split_fasta_parallel(input_fasta, tmp_dir, threads)
        
        ## fasta to int-encoded
        utils.fasta_int_encoded_parellel(tmp_dir, threads, min_length)

        ## convert int-encoded csv to kmerfre array and onehot-encoded array
        ## the resultant arrays could be used for prediction
        utils.save_npz_parellel(tmp_dir, threads)

    def preprocessing_bin_dir(bin_fasta, tmp_dir, min_length, threads):
        utils.preprocessing_bin_parellel(bin_fasta, tmp_dir, min_length, threads)


    # prediction for inidividual fragment
    def prediction(tmp_dir):
        indexs = [f.split("forward_")[1].split(".npz")[0] for f in glob.glob(f"{tmp_dir}/forward*.npz")]
        for index in indexs:
            utils.predict(tmp_dir, index)


    # organize fragment prediction results and generate final output
    def generate_final_output(tmp_dir):
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
        stage1_confidence_dic = {}
        for f in set(stage1_res.origin):
            df = stage1_res.query('origin == @f')
            # if the input sequence unfragmented (seq length < 10,000 bp)
            if len(df) == 1:
                prediction = list(df.predict)[0]
                stage1_final_prediction[f] = prediction
                stage1_confidence_dic[f] = 1
            # if the input sequence fragmented, the final prediction for the original sequence determined based on the majority rule
            else:
                prediction = list(df.predict)
                prediction_pro = prediction.count("prokaryotes")
                prediction_euk = prediction.count("eukaryotes")
                stage1_confidence = max(prediction_euk,prediction_pro) / (prediction_pro + prediction_euk)
                stage1_confidence_dic[f] = stage1_confidence
                if stage1_confidence > s1: 
                    if prediction_pro > prediction_euk:
                        stage1_final_prediction[f] = "prokaryotes"
                    else:
                        stage1_final_prediction[f] = "eukaryotes"
                else:
                    stage1_final_prediction[f] = "undetermined"

        stage2_final_prediction = {}
        stage2_confidence_dic = {}
        for f in set(stage2_res.origin):
            df = stage2_res.query('origin == @f')
            if len(df) == 1:
                prediction = list(df.predict)[0]
                stage2_final_prediction[f] = prediction
                stage2_confidence_dic[f] = 1
            else:
                prediction = list(df.predict)
                prediction_fungi = prediction.count("fungi")
                prediction_protozoa = prediction.count("protozoa") 
                stage2_confidence = max(prediction_protozoa,prediction_fungi) / (prediction_protozoa + prediction_fungi)
                stage2_confidence_dic[f] = stage2_confidence
                if stage2_confidence > s2:
                    if prediction_fungi > prediction_protozoa:
                        stage2_final_prediction[f] = "fungi"
                    else:
                        stage2_final_prediction[f] = "protozoa"                       
                else:
                    stage2_final_prediction[f] = "undetermined"
                    
        
        stage1_final_prediction_df = pd.DataFrame.from_dict(stage1_final_prediction, orient = "index").rename(columns = {0:"stage1_prediction"})
        stage2_final_prediction_df = pd.DataFrame.from_dict(stage2_final_prediction, orient = "index").rename(columns = {0:"stage2_prediction"})
        stage1_confidence_df = pd.DataFrame.from_dict(stage1_confidence_dic, orient = "index").rename(columns = {0:"stage1_confidence"})
        stage2_confidence_df = pd.DataFrame.from_dict(stage2_confidence_dic, orient = "index").rename(columns = {0:"stage2_confidence"})
        final_output_tmp1 = pd.merge(stage1_final_prediction_df, stage2_final_prediction_df, left_index = True, right_index = True, how="left")
        final_output_tmp2 = pd.merge(final_output_tmp1, stage1_confidence_df, left_index = True, right_index = True, how="left")
        final_output = pd.merge(final_output_tmp2, stage2_confidence_df, left_index = True, right_index = True, how="left")
        final_output.reset_index(names = "sequence_id", inplace = True)
        final_output.stage2_prediction = final_output.stage2_prediction.fillna("prokaryotes")
        final_output.loc[list(final_output.query("stage1_prediction == 'prokaryotes'").index), "stage2_prediction"] = "prokaryotes"
        final_output.loc[list(final_output.query("stage1_prediction == 'prokaryotes'").index), "stage2_confidence"] = 0
        final_output.loc[list(final_output.query("stage1_prediction == 'undetermined'").index), "stage2_prediction"] = "undetermined"
        final_output.loc[list(final_output.query("stage1_prediction == 'undetermined'").index), "stage2_confidence"] = 0
        return final_output

    def generate_final_output_for_bins(tmp_dir):
        bin_list = []
        stage1_predict = []
        stage2_predict = []
        stage1_confidence = []
        stage2_confidence = []
        for _ in glob.glob(f"{tmp_dir}/*"):
            Bin = _.split("/")[-1]
            bin_list.append(Bin)
            stage1_df_list = []
            stage2_df_list = []

            for stage1_out in glob.glob(f"{tmp_dir}/{Bin}/*stage1*.csv"):
                stage1_df_list.append(pd.read_csv(stage1_out))
                stage1_df = pd.concat(stage1_df_list)
            if len(stage1_df) == 0:
                logging.info(f"{Bin} does not have any contig that is longer than the minimal contig length")
                stage1_predict.append("NA")
                stage1_confidence.append("NA")
            else:
                eukaryotes_percent = len(stage1_df.query('predict == "eukaryotes"'))/len(stage1_df)
                prokaryotes_percent = 1 - eukaryotes_percent
                if eukaryotes_percent > prokaryotes_percent:
                    if eukaryotes_percent > s1:
                        stage1_predict.append("eukaryotes")
                        stage1_confidence.append(eukaryotes_percent)
                    else:
                        stage1_predict.append("undetermined")
                        stage1_confidence.append("NA")

                elif eukaryotes_percent == prokaryotes_percent:
                    stage1_predict.append("undetermined")
                    stage1_confidence.append("NA")

                else:
                    if prokaryotes_percent > s1:
                        stage1_predict.append("prokaryotes")
                        stage1_confidence.append(prokaryotes_percent)
                    else:
                        stage1_predict.append("undetermined")
                        stage1_confidence.append("NA")


            if stage1_predict[-1] != "eukaryotes":
                stage2_predict.append("NA")
                stage2_confidence.append("NA")

            else:
                for stage2_out in glob.glob(f"{tmp_dir}/{Bin}/*stage2*.csv"):
                    stage2_df_list.append(pd.read_csv(stage2_out))
                    stage2_df = pd.concat(stage2_df_list)
                if len(stage2_df) == 0:
                    stage2_predict.append("NA")
                    stage2_confidence.append("NA")
                
                else:
                    fungi_percent = len(stage2_df.query('predict == "fungi"'))/len(stage1_df)
                    protozoa_percent = 1 - fungi_percent

                    if fungi_percent > protozoa_percent:
                        if fungi_percent > s2:
                            stage2_predict.append("fungi")
                            stage2_confidence.append(fungi_percent)
                        else:
                            stage2_predict.append("undetermined")
                            stage2_confidence.append("NA")

                    elif fungi_percent == protozoa_percent:
                        stage2_predict.append("undetermined")
                        stage2_confidence.append("NA")

                    else:
                        if protozoa_percent > s2:
                            stage2_predict.append("protozoa")
                            stage2_confidence.append(protozoa_percent)
                        else:
                            stage2_predict.append("undetermined")
                            stage2_confidence.append("NA")

        bin_predict_out = pd.DataFrame.from_dict({"bin":bin_list, "stage1_prediction":stage1_predict, "stage1_confidence":stage1_confidence, "stage2_prediction":stage2_predict, "stage2_confidence":stage2_confidence })
        return bin_predict_out

    
    def write_to_fasta(final_output):
        prokaryotes = list(final_output.query('stage1_prediction == "prokaryotes"').sequence_id)
        eukaryotes = list(final_output.query('stage1_prediction == "eukaryotes"').sequence_id)
        protozoa = list(final_output.query('stage2_prediction == "protozoa"').sequence_id)
        fungi = list(final_output.query('stage2_prediction == "fungi"').sequence_id)

        with open(f"{output_dir}/{fasta_filename_trailing_removed}_GutEuk_prokaryotes.fasta", "w") as prokaryotes_out:
            with open(f"{output_dir}/{fasta_filename_trailing_removed}_GutEuk_eukaryotes.fasta", "w") as eukaryotes_out:
                with open(f"{output_dir}/{fasta_filename_trailing_removed}_GutEuk_protozoa.fasta", "w") as protozoa_out:
                    with open(f"{output_dir}/{fasta_filename_trailing_removed}_GutEuk_fungi.fasta", "w") as fungi_out:
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
    if input_bin:
        # preprocessing/formating 
        Bins = glob.glob(f"{input_fasta}/*.fa") + glob.glob(f"{input_fasta}/*.fasta") + glob.glob(f"{input_fasta}/*.fna")
        for Bin in Bins:
            bin_basename = re.search(r"(.*).(fa|fasta|fna)$", Bin.split("/")[-1]).group(1)
            Bin = re.search(r"(.*).(fa|fasta|fna)$", Bin).group(1) + ".fasta"
            try:
                os.mkdir(f"{tmp_dir}/{bin_basename}")
            except FileExistsError:
                pass

        preprocessing_bin_dir(Bins, tmp_dir, min_length, threads)
        preprocessing_end = time.time()
        logging.info(f"Preprocessing finished in {preprocessing_end - preprocessing_start:.2f} secs")

        # prediction
        prediction_start = time.time()
        for bin_dir in glob.glob(f"{tmp_dir}/*"):
            utils.prediction_bin(bin_dir)
        bin_level_predict_out = generate_final_output_for_bins(tmp_dir)
        bin_level_predict_out.to_csv(f"{output_dir}/{fasta_filename}_GutEuk_output.csv", index = None) 

    else:
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

        preprocessing(input_fasta, tmp_dir, threads, min_length)
        preprocessing_end = time.time()
        logging.info(f"Preprocessing finished in {preprocessing_end - preprocessing_start:.2f} secs")
        
        prediction_start = time.time()
        prediction(tmp_dir)
        final_output = generate_final_output(tmp_dir)
        final_output.to_csv(f"{output_dir}/{fasta_filename_trailing_removed}_GutEuk_output.csv", index = None)
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