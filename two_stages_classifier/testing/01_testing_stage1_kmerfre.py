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
import sys
sys.path.append('../training/')
import utils


## get prediction of kmerfre to train the ensembled model.

# Setup path to save model state_dict
path = "../../GutEuk/model/stage1-kmerfre.pth"
model_path = Path(path)

device = "cuda" if torch.cuda.is_available() else "cpu"

model = utils.kmerfre(1)
checkpoint = torch.load(model_path)
model.load_state_dict(checkpoint['model'])
model.to(device)

# Put the model in evaluation mode
model.eval()
with torch.inference_mode():
    for index in tqdm(range(80), desc = "Batchs (validation) finished", miniters=4):
        prediction = []
        test = utils.mydataset_m2("/fs/scratch/PAS0439/Ming/databases/gut_eukaryotes_classifier/transformed_dataset/test", str(index).zfill(3), 1)   
        test_loader = DataLoader(dataset=test,
                      shuffle=False,
                      num_workers=20)   
        for i, (forward, ID, kmerfre, y) in enumerate(test_loader):
            kmerfre = kmerfre.view(1, 1, -1, 1).to(torch.float32).to(device)
            # 1. Forward pass
            y_ce = model(kmerfre) 
            y_pred = torch.softmax(y_ce, dim=1)
            prediction.append(y_pred.numpy()[0])
            
        np.savez(f"/fs/ess/PAS0439/MING/cilates_fungi_classifier/outputs/GutEuk_classifier/stage1/testing/pred_kmerfre_{str(index).zfill(3)}.npz" , *prediction)
