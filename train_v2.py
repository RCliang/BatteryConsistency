import math
import copy
import os
import time
import enum
import argparse
import polars as pl
# Visualization related imports
import matplotlib.pyplot as plt
import seaborn
from torch.utils.data import DataLoader, Dataset
# Deep learning related imports
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.optim import Adam
from torch.utils.tensorboard import SummaryWriter
from torch.hub import download_url_to_file
from torch.utils.data import Dataset
# Data manipulation related imports
# from torchtext.data import Dataset, BucketIterator, Field, Example
import spacy
import seaborn as sns
import pandas as pd
import numpy as np
import datetime
import optuna
from DTN_model import DTN_model
from utils import processor,BatteryData

FINAL_COLS = [f"D135_{i}_dv" for i in range(1,113)]+['D135_diff','D135_std_max']+['cycle','samplingtimesec']
dataProcess = processor(['cycle'], 10, 5)

def train_loop(data_dir, model, loss_fn, optimizer, batch_size):
    size = 0
    # Set the model to training mode - important for batch normalization and dropout layers
    # Unnecessary in this situation but added for best practices
    model.train()
    train_loss=0
    batch = 0
    for file in [x for x in os.listdir(data_dir) if 'parquet' in x]:
        print(os.path.join(data_dir,file))
        # key = file.split('_')[0]
        battery_data = BatteryData(os.path.join(data_dir,file), dataProcess, FINAL_COLS)
        dataloader = DataLoader(dataset=battery_data,batch_size=batch_size,shuffle=True,num_workers=0)
        size += len(dataloader.dataset)
        # loss_file[key] = []
        tmp_loss=[]
        for _batch, (X, y) in enumerate(dataloader):
            pred = model(X.float().cuda(), model.get_key_padding_mask(X.cuda()).float().cuda())

            loss = loss_fn(pred.reshape((X.shape[0],X.shape[1],-1)).float(), y.float().cuda())
            train_loss+=loss.item()
            tmp_loss.append(loss.item())
            # Backpropagation
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            batch+=1
            if batch % 100 == 0:
                loss, current = loss.item(), (batch + 1) * len(X)
                print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")
                # loss_train.append(sum(tmp_loss)/len(tmp_loss))
                tmp_loss=[]
    return train_loss

def train(number_of_heads, number_of_layers, learning_rate):
    MyModel = DTN_model(512, number_of_heads, number_of_layers, 512, 0.3, 114, [512,64,16,1], 10, 10, 10, 2)
    optimizer = Adam(MyModel.parameters(),lr=learning_rate, betas=(0.9, 0.98), eps=1e-9)
    MyModel = MyModel.cuda()
    loss = nn.MSELoss()
    loss=loss.cuda()
    epochs = 20
    batch_size=4
    all_losses=[]
    for epoch in range(1,epochs+1):
        train_loss = train_loop('E:/projects/processed_data/normal_sample/charge_data',MyModel,loss,optimizer,batch_size)
        all_losses.append(train_loss)
        if epoch % 20 == 0:
            torch.save({
                    'epoch': epoch,
                    'model_state_dict': MyModel.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': loss,
                    }, f"model/10_5_moredata_model_charge_epoch{epoch}.pt")
    return all_losses

def objective(trail):
    # batchsize = trail.suggest_int('batchsize', 1, 16)
    lr = trail.suggest_float('lr', 1e-4, 1e-2,step=0.0001)
    # lossfunc = trail.suggest_categorical('loss', ['MSE', 'MAE'])
    # opt = trail.suggest_categorical('opt', ['Adam', 'SGD'])
    number_of_heads = trail.suggest_int('number_of_heads', 4, 8, step=4)
    number_of_layers = trail.suggest_int('number_of_layers', 2,5, step=1)
    # activefunc = trail.suggest_categorical('active', ['relu', 'sigmoid', 'tanh'])
    # weightdekey = trail.suggest_float('weight_dekay', 0, 1,step=0.01)
    # momentum= trail.suggest_float('momentum',0,1,step=0.01)
    loss = train(number_of_heads, number_of_layers, lr)
    return loss

if __name__ == "__main__":
    st=time.time()
    study = optuna.create_study(study_name='DTN_loss', direction='minimize')
    study.optimize(objective, n_trials=20)
    print(study.best_params)
    print(study.best_trial)
    print(study.best_trial.value)
    print(time.time()-st)
    optuna.visualization.plot_param_importances(study).show()
    optuna.visualization.plot_optimization_history(study).show()
    optuna.visualization.plot_slice(study).show()