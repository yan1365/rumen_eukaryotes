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


# prediction
class mydataset(Dataset):

    def __init__(self, npz_path):
        self.npz_path = npz_path
        self.path = str(Path(self.npz_path))
        self.forward = glob.glob(f"{self.path}/forward*")
        self.forward.sort()
        self.ids = glob.glob(f"{self.path}/ids*")
        self.ids.sort()
        self.kmerfre = glob.glob(f"{self.path}/kmerfre*")
        self.kmerfre.sort()
        self.y = glob.glob(f"{self.path}/y*")
        self.y.sort()
        self.chunks = len(self.y)

        
    def __getitem__(self, index):
        forward_npz = np.load(self.forward[index])
        ids_npz = np.load(self.ids[index], allow_pickle=True)
        kmerfre_npz = np.load(self.kmerfre[index])
        y_npz = np.load(self.y[index])
        
        forward_list = []
        ids_list = []
        kmerfre_list = []
        y_list = []
        
        for i in forward_npz.files:
            forward_list.append(forward_npz[i])
        for i in ids_npz.files:
            ids_list.append(ids_npz[i])
        for i in kmerfre_npz.files:
            kmerfre_list.append(kmerfre_npz[i])
        for i in y_npz.files:
            y_list.append(y_npz[i])
            
        forward = np.vstack(forward_list)   
        ids = list(np.concatenate(ids_list))
        kmerfre = np.vstack(kmerfre_list)
        y = np.vstack(y_list)   
        
        return forward, ids, kmerfre, y
        
    def __len__(self):
        return self.chunks


class mydataset_m2(Dataset):

    def __init__(self, npz_path, index, BATCH):
        self.npz_path = npz_path
        self.index = index
        self.BATCH = BATCH
        self.path = str(Path(self.npz_path))
        self.forward = f"{self.path}/forward_{self.index}.npz"
        self.ids = f"{self.path}/ids_{self.index}.npz"
        self.kmerfre = f"{self.path}/kmerfre_{self.index}.npz"
        self.y = f"{self.path}/y_{self.index}.npz"
        
        # preprocess
        forward_npz = np.load(self.forward)
        ids_npz = np.load(self.ids, allow_pickle=True)
        kmerfre_npz = np.load(self.kmerfre)
        y_npz = np.load(self.y)
        
        forward_list = []
        ids_list = []
        kmerfre_list = []
        y_list = []
        
        for i in forward_npz.files:
            forward_list.append(forward_npz[i])
        for i in ids_npz.files:
            ids_list.append(ids_npz[i])
        for i in kmerfre_npz.files:
            kmerfre_list.append(kmerfre_npz[i])
        for i in y_npz.files:
            y_list.append(y_npz[i])
            
        self.forward = np.vstack(forward_list)   
        self.ids = list(np.concatenate(ids_list))
        self.kmerfre = np.vstack(kmerfre_list)
        self.y = np.vstack(y_list)   

        self.chunks = self.y.shape[0]//self.BATCH
        
        
    def __getitem__(self, index):
        return self.forward[self.BATCH*index:self.BATCH*(index+1),:,:] , self.ids[self.BATCH*index:self.BATCH*(index+1)], self.kmerfre[self.BATCH*index:self.BATCH*(index+1),:], self.y[self.BATCH*index:self.BATCH*(index+1),:]
        
    def __len__(self):
        return self.chunks


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

    