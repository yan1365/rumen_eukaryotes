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
import os, glob, re
import utils
from utils import precision_recall


# setup checkpoint
def checkpoint(model, filename):
    torch.save({
    'optimizer': optimizer.state_dict(),
    'model': model.state_dict(),
    }, filename)

def resume(model, optimizer, filename):
    checkpoint = torch.load(filename)
    model.load_state_dict(checkpoint['model'])
    optimizer.load_state_dict(checkpoint['optimizer'])

# Setup path to save model state_dict
path = "/fs/ess/PAS0439/MING/cilates_fungi_classifier/outputs/GutEuk_classifier/stage2/cnn_v4_dropout0.5_wd0.001/"
model_path = Path(path)
device = "cuda" if torch.cuda.is_available() else "cpu"

# get the earlystopping status, number of epochs val loss did not improve (counter) and the latest minimal validation loss (min_validation_loss) 
with open(f"{model_path}/early_stopping_status.txt", "r") as handle:
    counter = int(handle.readline().split(" ")[1].strip())
    min_validation_loss = float(handle.readline().split(" ")[1])

#hyper-parameter
num_epochs = 10
Patience = 5
Min_delta = 0
BATCH = 512
early_stopper = utils.EarlyStopper(patience = Patience, min_delta = Min_delta, counter = counter, min_validation_loss = min_validation_loss)  # resume early-stopping status

# initiate old_val_loss for checkpoint update
old_val_loss = min_validation_loss


# load model parameters and resume training
# get number of epochs finished (last_finished_epoch) and get the latest model parameters 
model = utils.cnn_v4(BATCH)
model.to(device)
criterion  = nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(model.parameters(), weight_decay=0.001, lr=0.00008) 

pth = glob.glob(f"{str(model_path)}/epoch-*.pth")
if len(pth) == 1:
    last_finished_epoch = re.search(r".*epoch-([0-9]{3}).pth", pth[0]).group(1)
    resume(model, optimizer, pth[0])

else:  # in case multiple pth may exists (val loss did not improve)
    saved_epoch_list = []
    for i in pth:
        saved_epoch = re.search(r".*epoch-([0-9]{3}).pth", i).group(1)
        saved_epoch_list.append(saved_epoch)
    saved_epoch_list.sort()
    last_finished_epoch = saved_epoch_list[-1]
    resume(model, optimizer, f"{str(model_path)}/epoch-{last_finished_epoch}.pth") 

print(f"resume from epoch {last_finished_epoch}")


epoch_list = []
train_loss_list = []
train_acc_list = []
val_loss_list = []
val_acc_list = []
outputdic = {}


# for precision and recall
y_list_train = []
y_list_val = []

y_pred_list_train = []
y_pred_list_val = []

for epoch in tqdm(range(num_epochs), desc = "Epochs finished"):
    train_loss , train_acc = 0, 0 
    start = time.time()
    train_batches = 0
    for index in tqdm(range(80), desc = "Batchs (training) finished", miniters=16): 
        train = utils.mydataset_m2("/fs/scratch/PAS0439/Ming/databases/gut_eukaryotes_classifier/stage2/transformed_dataset/train", str(index).zfill(3), BATCH)
        
        train_loader = DataLoader(dataset=train,
                          shuffle=True,
                          num_workers=28)

        
        
    
        for i, (forward, _, kmerfre, y) in enumerate(train_loader):
            train_batches += 1
            dna_forward = forward.view(BATCH, 1, 5000, 4).to(torch.float32).to(device)
            kmerfre = kmerfre.view(BATCH, 1, -1, 1).to(torch.float32).to(device)
            y = y.view(BATCH).to(device)

            # Forward pass
            outputs = model(dna_forward)
            loss = criterion(outputs, y)
            train_loss += loss.item()
            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # calculate accuracy
            train_predicted = torch.softmax(outputs, dim=1).argmax(dim=1)
            train_accuray = accuracy(train_predicted, y, task="binary").item()
            train_acc += train_accuray

            y_list_train.append(y.numpy().flatten().tolist())
            y_pred_list_train.append(train_predicted.detach().numpy().flatten().tolist())
            

    # cal preci and recall for both classes
    y_list = []
    y_pred_list = []
    for f in range(len(y_list_train)):
        y_list += y_list_train[f]

    for f in range(len(y_pred_list_train)):
        y_pred_list += y_pred_list_train[f]
    
    precision_recall(y_pred_list, y_list)


    train_loss /= train_batches
    train_acc /= train_batches
    train_acc = train_acc*100

    
    
    # Put the model in evaluation mode
    model.eval()
    val_loss , val_acc = 0, 0 
    val_batches = 0
    with torch.inference_mode():
        for index in tqdm(range(40), desc = "Batchs (validation) finished", miniters=4): 
            val = utils.mydataset_m2("/fs/scratch/PAS0439/Ming/databases/gut_eukaryotes_classifier/stage2/transformed_dataset/val", str(index).zfill(3), BATCH)   
            val_loader = DataLoader(dataset=val,
                          shuffle=False,
                          num_workers=28)   
            for i, (forward, _, kmerfre, y) in enumerate(val_loader):
                dna_forward = forward.view(BATCH, 1, 5000, 4).to(torch.float32).to(device)
                kmerfre = kmerfre.view(BATCH, 1, -1, 1).to(torch.float32).to(device)
                y = y.view(BATCH).to(device)
                
                # 1. Forward pass
                y_ce = model(dna_forward) 
                y_pred = torch.softmax(y_ce, dim=1).argmax(dim=1)

                # 2. Calculate loss (accumatively)
                LOSS = F.cross_entropy(y_ce, y).item()
                val_loss =  val_loss + LOSS  

                # 3. Calculate accuracy 
                ACCURACY = accuracy(y_pred, y, task="binary").item()
                val_acc += ACCURACY
                val_batches += 1 
                
                y_list_val.append(y.numpy().flatten().tolist())
                y_pred_list_val.append(y_pred.detach().numpy().flatten().tolist())
    
    # cal preci and recall for both classes
    y_list = []
    y_pred_list = []
    for f in range(len(y_list_val)):
        y_list += y_list_val[f]

    for f in range(len(y_pred_list_val)):
        y_pred_list += y_pred_list_val[f]
    
    print("\n")
    precision_recall(y_pred_list, y_list)
                
        
            
    # Divide total test loss by length of test dataloader (per batch)
    val_loss /= val_batches
    # Divide total accuracy by length of test dataloader (per batch)
    val_acc /= val_batches
    val_acc = val_acc*100
    end = time.time()
    
    print(f"\nTrain loss: {loss:.5f} | Val loss: {val_loss:.5f} | Train acc: {train_acc:.2f}% | Val acc: {val_acc:.2f}% | {end - start:.2f} secs/epoch")
    
    # compare and save checkpoint

    if val_loss < early_stopper.min_validation_loss:  # if loss improved, delete old parameters. 
        print("Val loss improved, parameters updated!")
        checkpoint(model, f"{model_path}/epoch-{str(int(last_finished_epoch) + epoch + 1).zfill(3)}.pth")
        for f in list(range(epoch)):
            try: 
                #os.remove(f"{model_path}/epoch-{str(int(last_finished_epoch) + f + 1).zfill(3)}.pth")
                print(f"Old pth epoch-{str(int(last_finished_epoch) + f + 1).zfill(3)}.pth removed")
            except FileNotFoundError:
                continue
        old_val_loss = val_loss
    else:
        checkpoint(model, f"{model_path}/epoch-{str(int(last_finished_epoch) + epoch + 1).zfill(3)}.pth")
        print("Val loss did not improve...")
    
    # save output
    epoch_list.append(epoch+1)
    train_loss_list.append(train_loss)
    train_acc_list.append(train_acc)
    val_loss_list.append(val_loss)
    val_acc_list.append(val_acc)
    
    
    # early stopping
    if early_stopper.early_stop(val_loss):             
        break
    counter = early_stopper.counter
    min_validation_loss = early_stopper.min_validation_loss

    print(f"early stopping status| num epochs not improving: {counter}, min val loss: {min_validation_loss}")

outputdic["epoch"] = epoch_list
outputdic["train_loss"] = train_loss_list
outputdic["train_accuracy"] = train_acc_list
outputdic["val_loss"] = val_loss_list
outputdic["val_accuracy"] = val_acc_list
output_df = pd.DataFrame.from_dict(outputdic, orient = "columns")

finished_epoch = pd.read_csv(f"{path}/model_loss_accu.csv")
new_output_df = pd.concat([output_df, finished_epoch])

with open(f"{path}/early_stopping_status.txt", "w") as handle:
    handle.write(f"counter: {counter}\n")
    handle.write(f"min_validation_loss: {min_validation_loss}")

new_output_df.to_csv(f"{path}/model_loss_accu.csv", index = None)
print("Model performance updated!")