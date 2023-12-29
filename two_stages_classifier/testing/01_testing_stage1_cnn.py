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


device = "cuda" if torch.cuda.is_available() else "cpu"

path = "../../GutEuk/model/stage1-cnn.pth"
model_path = Path(path)

model = utils.cnn_v5(1)
checkpoint = torch.load(model_path)
model.load_state_dict(checkpoint['model'])
model.to(device)

# Put the model in evaluation mode
model.eval()
with torch.inference_mode():
    for index in tqdm(range(40), desc = "Batchs (validation) finished", miniters=4):
        prediction = []
        train = utils.mydataset_m2("/fs/scratch/PAS0439/Ming/databases/gut_eukaryotes_classifier/transformed_dataset/test", str(index).zfill(3), 1)   
        train_loader = DataLoader(dataset=train,
                      shuffle=False,
                      num_workers=20)   
        for i, (forward, ID, kmerfre, y) in enumerate(train_loader):
            dna_forward = forward.view(1, 1, 5000, 4).to(torch.float32).to(device)
            # 1. Forward pass
            y_ce = model(dna_forward) 
            y_pred = torch.softmax(y_ce, dim=1)
            prediction.append(y_pred.numpy()[0])
            
        np.savez(f"/fs/ess/PAS0439/MING/cilates_fungi_classifier/outputs/GutEuk_classifier/stage1/testing/pred_cnn_{str(index).zfill(3)}.npz" , *prediction)










