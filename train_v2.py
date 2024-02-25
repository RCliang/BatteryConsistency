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


class BatteryData(Dataset):
    def __init__(self, data_dir, processor, final_cols) -> None:
        super().__init__()
        self.processor = processor
        # self.ares = self.load_data(data_dir)
        data = pl.read_parquet(data_dir)
        self.ares, self.ares_y = self.processor.process_data(data.select(final_cols))

    def __len__(self):
        return self.ares.shape[0]
    
    def __getitem__(self, index):
        X = self.ares
        Y = self.ares_y
        return X[index], Y[index]
    
    def load_data(self, data_dir):
        path_list = os.listdir(data_dir)
        data = []
        for item in os.listdir(data_dir):
            data.append(self.processor.process_data(pl.read_parquet(os.path.join(data_dir,item))))
        res = pl.concat(data)
        return res
    
    @staticmethod
    def get_data_info(data_dir):
        data_info={}
        for root, dirs, files in os.walk(data_dir):
            for file in files:
                file_path = os.path.join(root, file)
                print(f"file: {file}")
                data = pl.read_parquet(file_path)
                yield data
class processor:
    def __init__(self, groups, seq_len, step_size) -> None:
        self.groups = groups
        self.seq_len = seq_len
        self.step_size = step_size
    
    def process_data(self, data:pl.DataFrame):
        final_res = []
        final_y = []
        for item in data.groupby(self.groups):
            data_tmp = item[1]
            data_tmp = data_tmp.drop('cycle')
            l = data_tmp.shape[0]
            if l < self.seq_len:
                continue
            # steps = int((l-self.seq_len)//self.step_size+1)
            steps = int(l-self.seq_len*2+self.step_size)
            for i in range(steps+1):
                X_tensor = torch.from_numpy(data_tmp[i:i+self.seq_len].to_numpy())
                y_tensor = torch.from_numpy(data_tmp[i+self.step_size:i+self.step_size+self.seq_len].select('D135_diff').to_numpy())
                final_res.append(X_tensor)
                final_y.append(y_tensor)
            # X_tensor = torch.from_numpy(data_tmp[-self.seq_len-self.step_size:-self.step_size].to_numpy())
            # if X_tensor.shape[0] < 10: continue
            # y_tensor = torch.from_numpy(data_tmp[-self.seq_len:].select('D135_diff').to_numpy())
            # final_res.append(X_tensor)
            # final_y.append(y_tensor)
        ares = torch.cat(final_res)
        ares = ares.reshape((-1, self.seq_len, data_tmp.shape[1]))
        ares_y = torch.cat(final_y)
        ares_y = ares_y.reshape((-1, self.seq_len, 1))
        return ares, ares_y

class MLP(nn.Module):
    def __init__(self, model_dimension):
        super(MLP, self).__init__()
        self.hidden1 = nn.Sequential(
            nn.Linear(
                in_features=model_dimension,
                out_features=256,
                bias=True
            ),
            nn.ReLU()
        )
        self.hidden2 = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU()
        )
        self.hidden3 = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU()
        )
        self.classifica = nn.Linear(64, 2)

    def forward(self, x):
        fc1 = F.dropout(self.hidden1(x))
        fc2 = F.dropout(self.hidden2(fc1))
        output = self.classifica(fc2)
        return F.log_softmax(output)


class PositionalEncoding(nn.Module):

    def __init__(self, model_dimension, dropout_probability, expected_max_sequence_length=5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout_probability)
        position_id = torch.arange(0, expected_max_sequence_length).unsqueeze(1)
        frequencies = torch.pow(10000., -torch.arange(0, model_dimension, 2, dtype=torch.float) / model_dimension)

        # Checkout playground.py for visualization of how these look like (it's super simple don't get scared)
        positional_encodings_table = torch.zeros(expected_max_sequence_length, model_dimension)
        positional_encodings_table[:, 0::2] = torch.sin(position_id * frequencies)  # sine on even positions
        positional_encodings_table[:, 1::2] = torch.cos(position_id * frequencies)  # cosine on odd positions

        self.register_buffer('positional_encodings_table', positional_encodings_table)

    def forward(self, embeddings_batch):
        assert embeddings_batch.ndim == 3 and embeddings_batch.shape[-1] == self.positional_encodings_table.shape[1], \
            f'Expected (batch size, max token sequence length, model dimension) got {embeddings_batch.shape}'

        positional_encodings = self.positional_encodings_table[:embeddings_batch.shape[1]]
        return self.dropout(embeddings_batch + positional_encodings)

class DTN_model(nn.Module):
    def __init__(self, model_dimension, number_of_heads, number_of_layers,d_hid, dropout_probability, number_of_var,
                 hidden_dimension_list,in_channel, out_channel, kernel_size, stride):
        super().__init__()
        # Periodic Embeddings
        # two different encoders and decoders
        EncoderLayer = nn.TransformerEncoderLayer(d_model=model_dimension,
                                                  nhead=number_of_heads,dim_feedforward=d_hid, batch_first=True)
        self.encoder = nn.TransformerEncoder(EncoderLayer, num_layers=number_of_layers)
        self.fw_embedding = nn.Linear(number_of_var, model_dimension)
        self.bw_embedding = nn.Linear(number_of_var, model_dimension)
        self.pos_embedding = PositionalEncoding(model_dimension, dropout_probability)

        self.decoder = nn.Conv1d(in_channel, out_channel, kernel_size, stride)
        self.mlp = nn.ModuleList([nn.Sequential(nn.Linear(int((model_dimension - kernel_size)/stride+1),hidden_dimension_list[0]),nn.ReLU())]
                                 +[nn.Sequential(nn.Linear(hidden_dimension_list[i-1],hidden_dimension_list[i]),nn.ReLU()) for i in range(1,len(hidden_dimension_list))])
        self.init_params()

    def init_params(self):
        # I tested both PyTorch's default initialization and this, and xavier has tremendous impact! I didn't expect
        # that the model's perf, with normalization layers, is so dependent on the choice of weight initialization.
        for name, p in self.named_parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    @staticmethod
    def get_key_padding_mask(tokens):
        tmp=tokens[:,:,0].squeeze()
        tmp = tmp.reshape((-1,tokens.shape[1]))
        key_padding_mask = torch.zeros(tmp.size())
        key_padding_mask[tmp == 0] = -torch.inf
        return key_padding_mask
    
    @staticmethod
    def get_bw_mask(bw_tensor):
        bw_mask = nn.Transformer.generate_square_subsequent_mask(bw_tensor.size()[-2])
        return bw_mask
    
    def encode(self, input, src_mask):
        input_embeddings = self.fw_embedding(input)
        input_embeddings = self.pos_embedding(input_embeddings)
        input_repr = self.encoder(input_embeddings, src_key_padding_mask=src_mask)
        return input_repr
    
    def decode(self, input):
        return bw_repr
        
    def forward(self, input, src_mask):
        input_repr = self.encode(input, src_mask)
        # decode_repr = self.decoder(input_repr.permute([0,2,1])).permute([0,2,1])
        decode_repr = self.decoder(input_repr)
        for m in self.mlp:
            decode_repr = m(decode_repr)
        # for m in self.mlp:
        #     input_repr = m(input_repr)
        # pred = F.log_softmax(input_repr,dim=1)
        return decode_repr


def train(batch_size, learning_rate, lossfunc, opt, hidden_layer, activefunc, weightdk,momentum):  # 选出一些超参数
    trainset_num = 800
    testset_num = 50

    train_dataset = myDataset(trainset_num)
    test_dataset = myDataset(testset_num)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)


    # 创建CNN模型， 并设置损失函数及优化器
    model = MLP(hidden_layer, activefunc).cuda()
    # print(model)
    if lossfunc == 'MSE':
        criterion = nn.MSELoss().cuda()
    elif lossfunc == 'MAE':
        criterion = nn.L1Loss()
  
    # optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weightdk)
    optimizer =optim.RMSprop(model.parameters(),lr=learning_rate,weight_decay=weightdk, momentum=momentum)
    # 训练过程
    for epoch in range(num_epoches):
        # 训练模式
        model.train()
        for i, data in enumerate(train_loader):
            inputs, labels, _ = data
            inputs = Variable(inputs).float().cuda()
            labels = Variable(labels).float().cuda()
            # 前向传播
            out = model(inputs)
            # 可以考虑加正则项
            train_loss = criterion(out, labels)
            optimizer.zero_grad()
            train_loss.backward()
            optimizer.step()

    model.eval()
  	
    testloss = test() #返回测试集合上的MAE
    print('Test MAE = ', resloss)
    return resloss

def objective(trail):
    batchsize = trail.suggest_int('batchsize', 1, 16)
    lr = trail.suggest_float('lr', 1e-4, 1e-2,step=0.0001)
    lossfunc = trail.suggest_categorical('loss', ['MSE', 'MAE'])
    opt = trail.suggest_categorical('opt', ['Adam', 'SGD'])
    hidden_layer = trail.suggest_int('hiddenlayer', 20, 1200)
    activefunc = trail.suggest_categorical('active', ['relu', 'sigmoid', 'tanh'])
    weightdekey = trail.suggest_float('weight_dekay', 0, 1,step=0.01)
    momentum= trail.suggest_float('momentum',0,1,step=0.01)
    loss = train(batchsize, lr, lossfunc, opt, hidden_layer, activefunc, weightdekey,momentum)
    return loss

if __name__ == '__main__':
    st=time.time()
    study = optuna.create_study(study_name='test', direction='minimize')
    study.optimize(objective, n_trials=500)
    print(study.best_params)
    print(study.best_trial)
    print(study.best_trial.value)
    print(time.time()-st)
    optuna.visualization.plot_param_importances(study).show()
    optuna.visualization.plot_optimization_history(study).show()
    optuna.visualization.plot_slice(study).show()


if __name__ == "__main__":
    model_dimension = 512
    number_of_heads = 4
    number_of_var = 114
    number_of_layers = 3
    dropout_probability = 0.4
    hidden_dimension_list = [512,64,16,1]
    d_hid = 512
    in_channel = 10
    out_channel = 10
    kernel_size = 10
    stride = 2

    dataProcess = processor(['cycle'], 10, 5)

    st=time.time()
    study = optuna.create_study(study_name='test', direction='minimize')
    study.optimize(objective, n_trials=500)
    print(study.best_params)
    print(study.best_trial)
    print(study.best_trial.value)
    print(time.time()-st)
    optuna.visualization.plot_param_importances(study).show()
    optuna.visualization.plot_optimization_history(study).show()
    optuna.visualization.plot_slice(study).show()