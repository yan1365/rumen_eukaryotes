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


# model architecture
class kmerfre(nn.Module):
    def __init__(self, BATCH):
        super().__init__()
        self.BATCH = BATCH
        

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
        x = self.fc(kmerfre.view(self.BATCH,-1))
        return x


class cnn(nn.Module):
    def __init__(self, BATCH):
        super().__init__()
        self.BATCH = BATCH
        
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

    