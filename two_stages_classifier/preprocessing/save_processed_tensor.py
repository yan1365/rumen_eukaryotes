#! /usr/bin/env python3
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import utils_preprocessing as utils
import os
import math
import pandas as pd
import subprocess
import multiprocessing
import sys
from pathlib import Path

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



   
def save_npz(input_file, index_list, outputdic, index, BATCH=128): # BATCH size to limit memory usage
    ids_list = []
    forward_list = []
    #reverse_list = []
    kmerfre_list = []
    y_list = []

    for f in index_list:
        df = next(pd.read_csv(input_file, chunksize = BATCH, header = None, skiprows=BATCH*f))
        ids = np.array(df.iloc[:,0])
        df = df.iloc[:,1:]
        x = np.array(df.iloc[:,:-1])
        y = np.array(df.iloc[:,-1]).reshape(-1,1)
        dna_forward = dna2onehot(x)
        #dna_reverse_compli = revcompli(dna_forward)
        kmerfre = seq2kmerfrequency(x)
        ids_list.append(ids)
        forward_list.append(dna_forward)
        #reverse_list.append(dna_reverse_compli)
        kmerfre_list.append(kmerfre)
        y_list.append(y)

    np.savez(f"{outputdic}/ids_{index}.npz" , *ids_list)    
    np.savez(f"{outputdic}/kmerfre_{index}.npz" , *kmerfre_list)
    np.savez(f"{outputdic}/forward_{index}.npz" , *forward_list)
    #np.savez(f"{outputdic}/reverse_{index}.npz" , *reverse_list)
    np.savez(f"{outputdic}/y_{index}.npz" , *y_list)
    print("saved")

def save_npz_wrapper(args):
        save_npz(*args)

ncores = os.cpu_count()    

def save_npz_parellel(args_list):
    
    with multiprocessing.Pool(processes=ncores) as pool:
        pool.map(save_npz_wrapper, args_list)

if __name__ == "__main__":
    BATCH = int(sys.argv[1])
    input_file = str(sys.argv[2])
    output_dir = str(sys.argv[3])
    output_path = str(Path(output_dir))
    cmd_count_line = f"cat {input_file}|wc -l"
    n_samples = int(subprocess.check_output(cmd_count_line, shell=True, text=True))
    n_batch = math.ceil(n_samples//BATCH)

    
    n_files = 40 # split large file into smaller chunks
    core_list = [[] for _ in range(n_files)]
    # each file split multiple batchs for memory and assign to multiple files to process in parallel

    for f in range(n_batch):
        i = f%n_files
        core_list[i].append(f)

    args_list = []    
    for f in range(n_files):
        args_list.append((input_file, core_list[f], output_path , str(f).zfill(3), BATCH)) 
        

    save_npz_parellel(args_list)