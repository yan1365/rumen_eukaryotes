import torch
import torchvision
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
import subprocess
import math
import time
from tqdm.auto import tqdm
from torchmetrics.functional import accuracy, f1_score, auroc
import random
from pathlib import Path
import os
import glob

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


class mydataset_ensembled(Dataset):

    def __init__(self, npz_path, index, BATCH):
        self.npz_path = npz_path
        self.index = index
        self.BATCH = BATCH
        self.path = str(Path(self.npz_path))
        self.cnn = f"{self.path}/pred_cnn_{self.index}.npz"
        self.ids = f"{self.path}/ids_{self.index}.npz"
        self.kmerfre = f"{self.path}/pred_kmerfre_{self.index}.npz"
        self.y = f"{self.path}/y_{self.index}.npz"
        
        # preprocess
        ids_npz = np.load(self.ids, allow_pickle=True)
        cnn_npz = np.load(self.cnn)        
        kmerfre_npz = np.load(self.kmerfre)
        y_npz = np.load(self.y)
        
        ids_list = []
        cnn_list = []
        kmerfre_list = []
        y_list = []
        
        for i in ids_npz.files:
            ids_list.append(ids_npz[i])
        for i in cnn_npz.files:
            cnn_list.append(cnn_npz[i])
        for i in kmerfre_npz.files:
            kmerfre_list.append(kmerfre_npz[i])
        for i in y_npz.files:
            y_list.append(y_npz[i])
            
        self.ids = list(np.concatenate(ids_list))
        self.cnn_out = np.vstack(cnn_list)   
        self.kmerfre_out = np.vstack(kmerfre_list)
        self.y = np.vstack(y_list)   

        self.chunks = self.y.shape[0]//self.BATCH
        
        
    def __getitem__(self, index):
        
        return self.ids[self.BATCH*index:self.BATCH*(index+1)], self.cnn_out[self.BATCH*index:self.BATCH*(index+1),:], self.kmerfre_out[self.BATCH*index:self.BATCH*(index+1),:], self.y[self.BATCH*index:self.BATCH*(index+1),:]
        
    def __len__(self):
        return self.chunks


# define model
# kmerfre only model
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

class guteuk(nn.Module):
    def __init__(self, BATCH):
        super().__init__()
        self.BATCH = BATCH
        
        self.conv = nn.Sequential(
            nn.Conv2d(1, 64, (6, 4)), #tune
            nn.ReLU(),
            nn.Flatten(start_dim=2),
            nn.MaxPool1d(3),  # tune
            nn.BatchNorm1d(64),
            nn.Conv1d(64, 128, 3), # tune
            nn.ReLU(),
            nn.MaxPool1d(3), # tune
            nn.BatchNorm1d(128),
            nn.Conv1d(128, 256, 3), #tune
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.AdaptiveMaxPool1d(1),
            nn.Flatten()
        )
        
        self.fc = nn.Sequential(
            nn.Linear(1280, 256),  # 2*256 + 1024 (kmerfre)  
            nn.ReLU(),
            nn.Dropout(p=0.1), # default 0.2
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Dropout(p=0.1), # default 0.2
            nn.Linear(64, 2)            
        )
        
  

    def forward(self, dna_forward, kmerfre):
        cnn_forward = self.conv(dna_forward)
        cnn_plus_kmer = torch.cat((cnn_forward.view(self.BATCH,-1), kmerfre.view(self.BATCH,-1)[:,4095:5119]), dim=1)
        
        x = self.fc(cnn_plus_kmer)
        return x

class guteuk_v2(nn.Module):
    def __init__(self, BATCH):
        super().__init__()
        self.BATCH = BATCH
        
        self.conv = nn.Sequential(
            nn.Conv2d(1, 64, (6, 4)), #tune
            nn.ReLU(),
            nn.Flatten(start_dim=2),
            nn.MaxPool1d(3),  # tune
            nn.BatchNorm1d(64),
            nn.Conv1d(64, 128, 3), # tune
            nn.ReLU(),
            nn.MaxPool1d(3), # tune
            nn.BatchNorm1d(128),
            nn.Conv1d(128, 256, 3), #tune
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.AdaptiveMaxPool1d(1),
            nn.Flatten()
        )
        
        self.fc = nn.Sequential(
            nn.Linear(5632, 1024),    
            nn.ReLU(),
            nn.Dropout(p=0.2), 
            nn.Linear(1024, 256),
            nn.ReLU(),
            nn.Dropout(p=0.2), 
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Dropout(p=0.2), 
            nn.Linear(64, 2)            
        )
        
  

    def forward(self, dna_forward, kmerfre):
        cnn_forward = self.conv(dna_forward)
        cnn_plus_kmer = torch.cat((cnn_forward.view(self.BATCH,-1), kmerfre.view(self.BATCH,-1)), dim=1)
        
        x = self.fc(cnn_plus_kmer)
        return x


class cnn(nn.Module):
    def __init__(self, BATCH):
        super().__init__()
        self.BATCH = BATCH
        
        self.conv = nn.Sequential(
            nn.Conv2d(1, 64, (6, 4)), #tune
            nn.ReLU(),
            nn.Flatten(start_dim=2),
            nn.MaxPool1d(3),  # tune
            nn.BatchNorm1d(64),
            nn.Conv1d(64, 128, 3), # tune
            nn.ReLU(),
            nn.MaxPool1d(3, ceil_mode = True), # tune
            nn.BatchNorm1d(128),
            nn.Conv1d(128, 256, 3), #tune
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten()
        )
        
        self.fc = nn.Sequential(
            nn.Linear(256, 64),    
            nn.ReLU(),
            nn.Dropout(p=0.5), # default 0.2
            nn.Linear(64, 8),
            nn.ReLU(),
            nn.Dropout(p=0.5), # default 0.2
            nn.Linear(8, 2)            
        ) 
        
  

    def forward(self, dna_forward):
        cnn_forward = self.conv(dna_forward)
        
        x = self.fc(cnn_forward)
        return x

class cnn_v2(nn.Module):
    def __init__(self, BATCH):
        super().__init__()
        self.BATCH = BATCH
        
        self.conv = nn.Sequential(
            nn.Conv2d(1, 64, (6, 4)), #tune
            nn.ReLU(),
            nn.Flatten(start_dim=2),
            nn.MaxPool1d(3),  # tune
            nn.BatchNorm1d(64),
            nn.Conv1d(64, 128, 3), # tune
            nn.ReLU(),
            nn.MaxPool1d(3, ceil_mode = True), # tune
            nn.BatchNorm1d(128),
            nn.Conv1d(128, 256, 3), #tune
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten()
        )
        
        self.fc = nn.Sequential(
            nn.Linear(256, 128),    
            nn.ReLU(),
            nn.Dropout(p=0.2), # default 0.2
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(p=0.2), # default 0.2
            nn.Linear(64, 2)            
        )
        
  

    def forward(self, dna_forward):
        cnn_forward = self.conv(dna_forward)        
        x = self.fc(cnn_forward)
        return x

class cnn_v3(nn.Module):
    def __init__(self, BATCH):
        super().__init__()
        self.BATCH = BATCH
        
        self.conv = nn.Sequential(
            nn.Conv2d(1, 32, (6, 4)), #tune
            nn.ReLU(),
            nn.Flatten(start_dim=2),
            nn.MaxPool1d(3),  # tune
            nn.BatchNorm1d(32),
            nn.Conv1d(32, 64, 3), # tune
            nn.ReLU(),
            nn.MaxPool1d(3, ceil_mode = True), # tune
            nn.BatchNorm1d(64),
            nn.Conv1d(64, 128, 3), #tune
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten()
        )
        
        self.fc = nn.Sequential(
            nn.Linear(128, 32),    
            nn.ReLU(),
            nn.Dropout(p=0.5), 
            nn.Linear(32, 8),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(8, 2)            
        )
        
  

    def forward(self, dna_forward):
        cnn_forward = self.conv(dna_forward)        
        x = self.fc(cnn_forward)
        return x

class cnn_v4(nn.Module):
    def __init__(self, BATCH):
        super().__init__()
        self.BATCH = BATCH
        
        self.conv = nn.Sequential(
            nn.Conv2d(1, 16, (6, 4)), #tune
            nn.ReLU(),
            nn.Flatten(start_dim=2),
            nn.Conv1d(16, 32, 3), 
            nn.ReLU(),
            nn.MaxPool1d(3),  # tune
            nn.BatchNorm1d(32),
            nn.Conv1d(32, 64, 3), # tune
            nn.ReLU(),
            nn.Conv1d(64, 128, 3), # tune
            nn.ReLU(),
            nn.MaxPool1d(3, ceil_mode = True),
            nn.BatchNorm1d(128),
            nn.Conv1d(128, 256, 3), 
            nn.ReLU(),
            nn.Conv1d(256, 512, 3), 
            nn.ReLU(),
            nn.BatchNorm1d(512),
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten()
        )
        
        self.fc = nn.Sequential(
            nn.Linear(512, 128),    
            nn.ReLU(),
            nn.Dropout(p=0.5), 
            nn.Linear(128, 32),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(32, 2)            
        )
        
  

    def forward(self, dna_forward):
        cnn_forward = self.conv(dna_forward)        
        x = self.fc(cnn_forward)
        return x


class cnn_v5(nn.Module):
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


class ae(nn.Module):
    def __init__(self, BATCH):
        super().__init__()
        self.BATCH = BATCH

        self.encoder = nn.Sequential(
            nn.Linear(5376, 1024),
            nn.ReLU(),
            nn.Dropout(p=0.5), 
            nn.Linear(1024, 256),
            nn.ReLU())
        self.decoder = nn.Sequential(
            nn.Linear(256, 1024),
            nn.ReLU(),
            nn.Dropout(p=0.5), 
            nn.Linear(1024, 5376),
            nn.ReLU())
            

    def forward(self, kmerfre):
        x = self.encoder(kmerfre.view(self.BATCH,-1))
        x = self.decoder(x)
        return x



class cnn_lstm(nn.Module):
    def __init__(self, BATCH):
        super().__init__()
        self.BATCH = BATCH
        
        self.conv = nn.Sequential(
            nn.Conv2d(1, 32, (6, 4)), #tune
            nn.ReLU(),
            nn.Flatten(start_dim=2),
            nn.MaxPool1d(3), 
            nn.MaxPool1d(3),  
            nn.BatchNorm1d(32)
            )
        
        self.lstm = nn.Sequential(
            nn.LSTM(input_size=32, hidden_size=64, num_layers=1, batch_first = True)
            )
        
        self.fc = nn.Sequential(
            nn.Linear(64, 8),    
            nn.ReLU(),
            nn.Dropout(p=0.2), # default 0.2  
            nn.Linear(8, 2)         
        )
        
  

    def forward(self, dna_forward):
        cnn_out = self.conv(dna_forward)
        cnn_out = cnn_out.permute(0, 2, 1)
        lstm_out, _ = self.lstm(cnn_out)
        x = self.fc(lstm_out[:,-1,:])
        return x
    
class cnn_lstm_v2(nn.Module):
    def __init__(self, BATCH):
        super().__init__()
        self.BATCH = BATCH
        
        self.conv = nn.Sequential(
            nn.Conv2d(1, 32, (6, 4)), #tune
            nn.ReLU(),
            nn.Flatten(start_dim=2),
            nn.MaxPool1d(3), 
            nn.MaxPool1d(3),  
            nn.BatchNorm1d(32)
            )
        
        self.lstm = nn.Sequential(
            nn.LSTM(input_size=32, hidden_size=128, num_layers=1, batch_first = True)
            )
        
        self.fc = nn.Sequential(
            nn.Linear(128, 24),    
            nn.ReLU(),
            nn.Dropout(p=0.2), # default 0.2  
            nn.Linear(24, 8),
            nn.ReLU(),     
            nn.Dropout(p=0.2), # default 0.2
            nn.Linear(8, 2)
        )
        
  

    def forward(self, dna_forward):
        cnn_out = self.conv(dna_forward)
        cnn_out = cnn_out.permute(0, 2, 1)
        lstm_out, _ = self.lstm(cnn_out)
        x = self.fc(lstm_out[:,-1,:])
        return x

class cnn_lstm_v3(nn.Module):
    def __init__(self, BATCH):
        super().__init__()
        self.BATCH = BATCH
        
        self.conv = nn.Sequential(
            nn.Conv2d(1, 32, (6, 4)), #tune
            nn.ReLU(),
            nn.Flatten(start_dim=2),
            nn.MaxPool1d(3), 
            nn.MaxPool1d(3),  
            nn.BatchNorm1d(32)
            )
        
        self.lstm = nn.Sequential(
            nn.LSTM(input_size=32, hidden_size=64, num_layers=1, batch_first = True, bidirectional=True)
            )
        
        self.fc = nn.Sequential(
            nn.Linear(128, 32),    
            nn.ReLU(),
            nn.Dropout(p=0.2), # default 0.2  
            nn.Linear(32, 8),
            nn.ReLU(),
            nn.Dropout(p=0.2), # default 0.2  
            nn.Linear(8, 2)          
        )
        
  

    def forward(self, dna_forward):
        cnn_out = self.conv(dna_forward)
        cnn_out = cnn_out.permute(0, 2, 1)
        lstm_out, _ = self.lstm(cnn_out)
        x = self.fc(lstm_out[:,-1,:])
        return x

# define earlystop
class EarlyStopper:
    def __init__(self, patience=10, min_delta=0, counter = 0, min_validation_loss = float('inf')):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = counter    # to continue training.
        self.min_validation_loss = float(min_validation_loss)

    def early_stop(self, validation_loss):
        if validation_loss < self.min_validation_loss:
            self.min_validation_loss = validation_loss
            self.counter = 0
        elif validation_loss > (self.min_validation_loss + self.min_delta):
            self.counter += 1
            if self.counter >= self.patience:
                return True
                print("Earlystopping !!!")
        return False

def precision_recall(y_pre:list, y: list):
    tp_1 = len(set(np.where(np.array(y_pre) == 1)[0]) & set(np.where(np.array(y) == 1)[0]))
    fp_1 = len(set(np.where(np.array(y_pre) == 1)[0]) & set(np.where(np.array(y) == 0)[0]))
    fn_1 = len(set(np.where(np.array(y_pre) == 0)[0]) & set(np.where(np.array(y) == 1)[0]))

    tp_0 = len(set(np.where(np.array(y_pre) == 0)[0]) & set(np.where(np.array(y) == 0)[0]))
    fp_0 = len(set(np.where(np.array(y_pre) == 0)[0]) & set(np.where(np.array(y) == 1)[0]))
    fn_0 = len(set(np.where(np.array(y_pre) == 1)[0]) & set(np.where(np.array(y) == 0)[0]))

    precision1 = tp_1/(tp_1 + fp_1) * 100
    precision0 = tp_0/(tp_0 + fp_0) * 100

    recall1 = tp_1/(tp_1 + fn_1) * 100
    recall0 = tp_0/(tp_0 + fn_0) * 100

    print(f"Class 0: Precision: {precision0:.2f}%| Recall: {recall0:.2f}%")
    print(f"Class 1: Precision: {precision1:.2f}%| Recall: {recall1:.2f}%")
    