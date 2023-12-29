#! /usr/bin/env python3
import numpy as np
import os
import math
import pandas as pd
import subprocess
import multiprocessing
import sys
from pathlib import Path
sys.path.append('/users/PAS1855/yan1365/rumen_eukaryotes/eukaryotes_classifier/model/preprocessing')
import utils_preprocessing as utils

# data transformation
def dna2onehot(X):
    forward = utils.one_hot_encode_with_zero_handling(X)
    return forward
    
def revcompli(foward):
    reverse_compli = utils.one_hot_reverse_compli(foward)
    return reverse_compli
    
def seq2kmerfrequency(X):
    inputs_list = []

    for f in range(len(X)):
        kmerfre = utils.kmerfrequency(X[f,:])
        inputs_list.append(kmerfre)

    inputs_kmerfre = np.array(inputs_list).reshape(len(X), -1)
    return inputs_kmerfre


   
def save_npz(input_file, index_list, outputdic, index, BATCH):
    forward_list = []
    reverse_list = []
    kmerfre_list = []
    y_list = []
    for f in index_list:
        df = next(pd.read_csv(input_file, chunksize = BATCH, header = None, skiprows=BATCH*f))
        df = df.iloc[:,1:]
        x = np.array(df.iloc[:,:-1])
        y = np.array(df.iloc[:,-1]).reshape(-1,1)
        dna_forward = dna2onehot(x)
        dna_reverse_compli = revcompli(dna_forward)
        kmerfre = seq2kmerfrequency(x)
        forward_list.append(dna_forward)
        reverse_list.append(dna_reverse_compli)
        kmerfre_list.append(kmerfre)
        y_list.append(y)
        
    np.savez(f"{outputdic}/kmerfre_{index}.npz" , *kmerfre_list)
    np.savez(f"{outputdic}/forward_{index}.npz" , *forward_list)
    np.savez(f"{outputdic}/reverse_{index}.npz" , *reverse_list)
    np.savez(f"{outputdic}/y_{index}.npz" , *y_list)
    print("saved")

def save_npz_wrapper(args):
        save_npz(*args)

ncores = os.cpu_count()    

def save_npz_parellel(args_list):
    
    with multiprocessing.Pool(processes=ncores) as pool:
        pool.map(save_npz_wrapper, args_list)

if __name__ == "__main__":
    BATCH = 128
    input_file = str(sys.argv[1])
    output_dir = str(sys.argv[2])
    output_path = str(Path(output_dir))
    cmd_count_line = f"cat {input_file}|wc -l"
    n_samples = int(subprocess.check_output(cmd_count_line, shell=True, text=True))
    ncores = os.cpu_count()   
    n_batch = math.ceil(n_samples//BATCH)   

    core_list = [[] for _ in range(ncores)]
    # each file split multiple batchs for memory and assign to multiple files to process in parallel

    for f in range(n_batch):
        i = f%ncores
        core_list[i].append(f)

    args_list = []    
    for f in range(ncores):
        args_list.append((input_file, core_list[f], output_path , str(f).zfill(3), BATCH)) 
        

    save_npz_parellel(args_list)