from typing import Any
from torch.utils.data import DataLoader, Dataset
from config import *
import polars as pl
import os
import torch
import torch.nn.functional as F



class processor:
    def __init__(self, groups, seq_len, step_size) -> None:
        self.groups = groups
        self.seq_len = seq_len
        self.step_size = step_size
    
    def process_data(self, data:pl.DataFrame):
        final_res = []
        for item in data.groupby(self.groups):
            data_tmp = item[1]
            l = data_tmp.shape[0]
            if l < self.seq_len:
                continue
            steps = int((l-self.seq_len)//self.step_size+1)
            for i in range(steps):
                X_tensor = torch.from_numpy(data_tmp[self.step_size*i:self.step_size*i+self.seq_len].to_numpy())
                final_res.append(X_tensor)
            X_tensor = torch.from_numpy(data_tmp[-self.seq_len:].to_numpy())
            final_res.append(X_tensor)
        ares = torch.cat(final_res)
        ares = ares.reshape((-1, self.seq_len, data.shape[1]))
        return ares
    

class BatteryData(Dataset):
    def __init__(self, data_dir, processor) -> None:
        super().__init__()
        self.processor = processor
        self.ares = self.processor.process_data(pl.read_parquet(data_dir))

    def __len__(self):
        return self.ares.shape[0]
    
    def __getitem__(self, index: Any) -> Any:
        X = self.ares[:, :, :-1]
        Y = self.ares[:, :, -1]
        return X[index], Y[index]
    
    @staticmethod
    def get_data_info(data_dir):
        data_info={}
        for root, dirs, files in os.walk(data_dir):
            for file in files:
                file_path = os.path.join(root, file)
                print(f"file: {file}")
                data = pl.read_parquet(file_path)
                yield data
    
    @staticmethod
    def get_data_info(data_dir):
        data_info={}
        for root, dirs, files in os.walk(data_dir):
            for file in files:
                file_path = os.path.join(root, file)
                print(f"file: {file}")
                data = pl.read_parquet(file_path)
                yield data
    


