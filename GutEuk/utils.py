import torch
import torchvision
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.nn.functional as F
import subprocess
import math
import time
from pathlib import Path
import os
import glob
from Bio import SeqIO
import numpy as np
import pandas as pd
import multiprocessing

# Author: Ming Yan, The Ohio State University

# preprocessing
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

def dna2onehot(X):
    forward = one_hot_encode_with_zero_handling(X)
    return forward
    
     
def seq2kmerfrequency(X):
    inputs_list = []

    for f in range(len(X)):
        kmerfre = kmerfrequency(X[f,:])
        inputs_list.append(kmerfre)

    inputs_kmerfre = np.array(inputs_list).reshape(len(X), -1)
    return inputs_kmerfre


def chop_sequence(seq, fragment_size = 5000, sliding_window = 5000):
    """Chop a sequence into fragments of equal size"""
    return  [seq[i:i+fragment_size] for i in range(0, len(seq), sliding_window)]
    

def split_fasta(input_fasta, seqlist, tmp_dir, index):
    """split sequences of a fasta file into files of similar sequence count"""
    
    with open(f"{tmp_dir}/input_fasta_int_encoded_{index}.fasta", "w") as outfile:
        records = SeqIO.parse(input_fasta, "fasta")
        for record in records:
            if str(record.id) in seqlist:
                SeqIO.write(record, outfile, "fasta")


def split_fasta_wrapper(args):
    split_fasta(*args)


def split_fasta_parallel(input_fasta, tmp_dir, threads):
    seqlist_to_assign = []
    total_seqs = []
    for f in range(threads):
        seqlist_to_assign.append([])

    records = SeqIO.parse(input_fasta, "fasta")
    for record in records:
        total_seqs.append(str(record.id))

    # assign each sequence in total_seqs to each thread
    for f in range(len(total_seqs)):
        thread = f % threads
        seqlist_to_assign[thread].append(total_seqs[f])   

    arg_list = [[input_fasta, seqlist_to_assign[f], tmp_dir, str(f+1).zfill(2)] for f in range(threads)]
   
    with multiprocessing.Pool(processes=threads) as pool:
        pool.map(split_fasta_wrapper, arg_list)

## fasta to int-encoded
def fasta_int_encoded(input_fasta_splited, tmp_dir, min_length):
    seq_origin = {}
    index = input_fasta_splited.split(".fasta")[0].split("_")[-1]
    with open(f"{tmp_dir}/input_fasta_int_encoded_{index}.csv", "w") as handle:
        records = SeqIO.parse(f"{tmp_dir}/{input_fasta_splited}", "fasta")
        for record in records:
            seqid = record.id
            seq = record.seq
            intencoded = dna2int(seq)
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

            seq_origin_df = pd.DataFrame.from_dict(seq_origin, orient = "index").reset_index()
            seq_origin_df.rename(columns = {"index": "seq", 0:'origin'}, inplace = True)
            seq_origin_df.to_csv(f"{tmp_dir}/seqorigin_{index}.csv", index = None)

def fasta_int_encoded_wrapper(args):
    fasta_int_encoded(*args)

def fasta_int_encoded_parellel(tmp_dir, threads, min_length):        
    int_encoded_fasta = glob.glob(f"{tmp_dir}/input_fasta_int_encoded*.fasta")
    if threads == 1:
        fasta_filename = os.path.basename(int_encoded_fasta[0])
        fasta_int_encoded(fasta_filename, tmp_dir, min_length)

    else:
        fasta_filename_list = [os.path.basename(f) for f in int_encoded_fasta]
        args_list = [[fasta_filename_list[f], tmp_dir, min_length] for f in range(len(fasta_filename_list))]

        with multiprocessing.Pool(processes=threads) as pool:
            pool.map(fasta_int_encoded_wrapper, args_list)

# int-encoded to npz
def int_encoded_to_array(int_encoded_csv, tmp_dir):  
    # convert int-encoded csv to kmerfre array and onehot-encoded array
    # the resultant arrays could be used for prediction
    df = pd.read_csv(int_encoded_csv, header = None)
    csv_filename = os.path.basename(int_encoded_csv)
    index = csv_filename.split(".csv")[0].split("_")[-1]
    ids = np.array(df.iloc[:,0])
    df = df.iloc[:,1:]
    x = np.array(df)
    dna_forward = dna2onehot(x)
    kmerfre = seq2kmerfrequency(x)

    np.savez(f"{tmp_dir}/ids_{index}.npz", *ids)    
    np.savez(f"{tmp_dir}/kmerfre_{index}.npz", *kmerfre)
    np.savez(f"{tmp_dir}/forward_{index}.npz", *dna_forward)

def int_encoded_to_array_wrapper(args):
    int_encoded_to_array(*args)

def save_npz_parellel(tmp_dir, threads):        
    int_encoded_fasta = glob.glob(f"{tmp_dir}/input_fasta_int_encoded*.csv")
    args_list = [[int_encoded_fasta[f], tmp_dir] for f in range(len(int_encoded_fasta))]
    with multiprocessing.Pool(processes=threads) as pool:
        pool.map(int_encoded_to_array_wrapper, args_list)

# load data
def load_npz(tmp_dir, index):

    forward_npz = np.load(f"{tmp_dir}/forward_{index}.npz")
    ids_npz = np.load(f"{tmp_dir}/ids_{index}.npz", allow_pickle=True)
    kmerfre_npz = np.load(f"{tmp_dir}/kmerfre_{index}.npz")
    
    forward_list = []
    ids_list = []
    kmerfre_list = []
    
    for i in forward_npz.files:
        forward_list.append(forward_npz[i])
    for i in ids_npz.files:
        ids_list.append(ids_npz[i])
    for i in kmerfre_npz.files:
        kmerfre_list.append(kmerfre_npz[i])
        

    forward_torch = torch.tensor(np.stack(forward_list, axis = 0)).to(torch.float32)   
    ids_np = np.stack(ids_list, axis = 0)
    kmerfre_torch = torch.tensor(np.stack(kmerfre_list, axis = 0)).to(torch.float32)

    return forward_torch, kmerfre_torch, ids_np

# model architecture
class model_kmerfre(nn.Module):
    def __init__(self):
        super().__init__()
        self.BATCH = 1
        

        self.fc = nn.Sequential(
            nn.Linear(5376, 1024),    
            nn.ReLU(),
            nn.Dropout(p=0.5), 
            nn.Linear(1024, 256),
            nn.ReLU(),
            nn.Dropout(p=0.5), 
            nn.Linear(256, 48),
            nn.ReLU(),
            nn.Dropout(p=0.5), 
            nn.Linear(48, 2)            
        )
        

    def forward(self, kmerfre):
        x = self.fc(kmerfre.view(1,-1))
        return x

class model_cnn(nn.Module):
    def __init__(self):
        super().__init__()
        self.BATCH = 1
        
        self.conv = nn.Sequential(
            nn.Conv2d(1, 32, (6, 4)), #tune
            nn.ReLU(),
            nn.Flatten(start_dim=2),
            nn.Conv1d(32, 64, 3), 
            nn.ReLU(),
            nn.MaxPool1d(3),  # tune
            nn.BatchNorm1d(64),
            nn.Conv1d(64, 128, 3), # tune
            nn.ReLU(),
            nn.Conv1d(128, 256, 3), # tune
            nn.ReLU(),
            nn.MaxPool1d(3, ceil_mode = True),
            nn.BatchNorm1d(256),
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten()
        )
        
        self.fc = nn.Sequential(
            nn.Linear(256, 64),    
            nn.ReLU(),
            nn.Dropout(p=0.5), 
            nn.Linear(64, 8),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(8, 2)            
        )

    def forward(self, dna_forward):
        cnn_forward = self.conv(dna_forward)        
        x = self.fc(cnn_forward)
        return x

# predict stage1 
def predict_stage1(forward_torch, kmerfre_torch, ids_np):

    prediction = []
    ID_list = []

    kmerfre = "model/stage1-kmerfre.pth"
    cnn = "model/stage1-cnn.pth"
    model_path_kmerfre = Path(kmerfre)
    model_path_cnn = Path(cnn)
    kmerfre = model_kmerfre()
    cnn = model_cnn()
    checkpoint_kmerfre = torch.load(model_path_kmerfre)
    checkpoint_cnn = torch.load(model_path_cnn)
    kmerfre.load_state_dict(checkpoint_kmerfre['model'])
    cnn.load_state_dict(checkpoint_cnn['model'])
    kmerfre.eval()
    cnn.eval()

    with torch.inference_mode():

        for f in range(len(ids_np)):
            forward_input = forward_torch[f].view(1, 1, 5000, 4)
            kmerfre_input = kmerfre_torch[f]
            ID = ids_np[f]
            
            y_ce_cnn = cnn(forward_input) 
            y_pred_cnn = torch.softmax(y_ce_cnn, dim=1)
            prediction_cnn = y_pred_cnn.detach().numpy()[0]
            
            y_ce_kmerfre = kmerfre(kmerfre_input) 
            y_pred_kmerfre = torch.softmax(y_ce_kmerfre, dim=1)
            prediction_kmerfre = y_pred_kmerfre.detach().numpy()[0]

            ID_list.append(ID)
            predict = (prediction_cnn + prediction_kmerfre)/2
            predict = predict.argmax().item()
            
            if predict == 0:
                prediction.append("prokaryotes")
            else:
                prediction.append("eukaryotes")
    
    stage1_res = pd.DataFrame.from_dict({"seq":ID_list, "predict":prediction})
    return stage1_res  


# predict stage2
def predict_stage2(forward_np, kmerfre_np, ids_np, index_proceed_stage2):

    prediction = []
    ID_list = []

    kmerfre = "model/stage2-kmerfre.pth"
    cnn = "model/stage2-cnn.pth"
    model_path_kmerfre = Path(kmerfre)
    model_path_cnn = Path(cnn)
    kmerfre = model_kmerfre()
    cnn = model_cnn()
    checkpoint_kmerfre = torch.load(model_path_kmerfre)
    checkpoint_cnn = torch.load(model_path_cnn)
    kmerfre.load_state_dict(checkpoint_kmerfre['model'])
    cnn.load_state_dict(checkpoint_cnn['model'])
    kmerfre.eval()
    cnn.eval()

    with torch.inference_mode():

        for f in index_proceed_stage2:
            forward_input = forward_np[f].view(1, 1, 5000, 4)
            kmerfre_input = kmerfre_np[f]
            ID = ids_np[f]
            
            # 1. Forward pass
            y_ce_cnn = cnn(forward_input) 
            y_pred_cnn = torch.softmax(y_ce_cnn, dim=1)
            prediction_cnn = y_pred_cnn.detach().numpy()[0]
            
            y_ce_kmerfre = kmerfre(kmerfre_input) 
            y_pred_kmerfre = torch.softmax(y_ce_kmerfre, dim=1)
            prediction_kmerfre = y_pred_kmerfre.detach().numpy()[0]

            ID_list.append(ID)
            predict = (prediction_cnn + prediction_kmerfre)/2
            predict = predict.argmax().item()
            
            if predict == 0:
                prediction.append("fungi")
            else:
                prediction.append("protozoa")
    
    stage2_res = pd.DataFrame.from_dict({"seq":ID_list, "predict":prediction})
    return stage2_res  

def predict(tmp_dir, index):
    forward_np, kmerfre_np, ids_np =  load_npz(tmp_dir, index)
    stage1_res = predict_stage1(forward_np, kmerfre_np, ids_np)
    index_proceed_stage2 = np.array(stage1_res.query('predict == "eukaryotes"').index)
    stage2_res = predict_stage2(forward_np, kmerfre_np, ids_np, index_proceed_stage2)
    stage1_res.to_csv(f"{tmp_dir}/input_fasta_{index}_stage1_out.csv", index = None)
    stage2_res.to_csv(f"{tmp_dir}/input_fasta_{index}_stage2_out.csv", index = None)


