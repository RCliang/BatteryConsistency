from typing import Any
from torch.utils.data import DataLoader, Dataset
from config import *
import polars as pl
import os
import torch
import torch.nn.functional as F


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