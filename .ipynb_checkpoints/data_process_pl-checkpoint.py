import os
import re
import sys
import time
import pickle
import logging
from tqdm import tqdm
import datetime
import gc
import pandas as pd
import polars as pl
import numpy as np
import math
import pytz
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")
import json
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, auc, roc_curve
from sklearn.metrics.pairwise import cosine_similarity
# import lightgbm as lgb
import torch
import torch.nn.functional as F
from config import *

output = "data/"
# raw_data_path = '/data/gaohe_dianxin/all_rawdata'
cols = ['cycle','D24_int','label']+[f"D135_{i}" for i in range(1,113)]+[f"D136_{i}" for i in range(1,28)]+['D27','soc_diff','D22','D356','D52','D334','D290']
seq_len=300
target_car_dict = {'D3':'2022-07-28',
                   'D2':'2022-05-20',
                   'D4':'2022-07-25',
                  'D1':'2022-03-14',
                  'D13':'2022-08-03',
                  'D440':'2022-09-06',
                  'D441':'2022-09-13',
                  'D442':'2022-11-21',
                  'D443':'2022-11-17'}

def get_file_path(raw_data_path, car_no, yu):
    res = []
    for x in os.listdir(raw_data_path):
        if (yu in x):
            if x.split("_")[1]==car_no:
                res.append(x)
    return [os.path.join(raw_data_path, file) for file in res]

def get_all_files(car_no, raw_data_path):
    vdcm_files = get_file_path(raw_data_path, car_no, "vdcm")
    ptcan_files = get_file_path(raw_data_path, car_no, "ptcan")
    bcan_files = get_file_path(raw_data_path, car_no, "bcan")
    bdcm_files = get_file_path(raw_data_path, car_no, "bdcm")
    return vdcm_files, ptcan_files, bcan_files, bdcm_files

def read_data(f, cols):
    try:
        if cols:
            df = pl.read_csv(f,dtypes={'samplingtimesec':str, 'D20':int,'D233':int, 'D19':int,'D356':int},columns=cols)
            return df
        else:
            return pl.read_csv(f,dtypes={'samplingtimesec':str, 'D20':int,'D233':int})
    except:
        return pl.DataFrame()
    
def concat_data(files, cols):
    vdcm=[]
    for f in files:
        df = read_data(f, cols)
        if df.shape[0]!=0:
            df = df.select(cols)
            vdcm.append(df)  
    print(f"files,{len(vdcm)}")
    res = pl.concat(vdcm)
    res = res.unique(subset='samplingtimesec', keep='first')
    return res

def step_1_process(car_no, raw_output_path):
    dfs={}
    vdcm_files, ptcan_files, bcan_files, bdcm_files = get_all_files(car_no)
    dfs['vdcm'] = concat_data(vdcm_files, vdcm_cols)
    print('vdcm end')
    dfs['ptcan'] = concat_data(ptcan_files, ptcan_cols)
    print('ptcan end')
    dfs['bcan'] = concat_data(bcan_files, bcan_cols)
    print('bcan end')
    dfs['bdcm'] = concat_data(bdcm_files, bdcm_cols)
    dfs['bcan'] = dfs['bcan'].drop_nulls(subset=['D140','D141'])
    dfs['bcan'] = dfs['bcan'].with_columns(pl.col("D141").apply(lambda x: re.search('(?<=:)\d+', x)[0] if type(x)!=int else x))
    dfs['bcan'] = dfs['bcan'].with_columns(pl.col("D140").apply(lambda x: re.search('(?<=:)\d+', x)[0] if type(x)!=int else x))
    dfs['ptcan'] = dfs['ptcan'].filter(pl.col('D19').is_between(2,5))
    df = dfs['bcan'].join(dfs['ptcan'], on='samplingtimesec', how='inner')
    del dfs['bcan'],dfs['ptcan']
    gc.collect()
    df = df.join(dfs['vdcm'].select(vdcm_cols), on='samplingtimesec', how='left')
    del dfs["vdcm"]
    gc.collect()
    df = df.join(dfs['bdcm'].select(bdcm_cols), on='samplingtimesec', how='left')
    del dfs['bdcm']
    gc.collect()
    try:
        df=df.drop(['samplingtime', 'tboxsendtimestamp', 'receivetimestamp'])
    except:
        pass
    df = df.with_columns(pl.col("samplingtimesec").cast(pl.Int64))
    df = df.sort("samplingtimesec")
    print(f"car{car_no} shape is {df.shape}")
    final_path = os.path.join(raw_output_path, f'{car_no}.parquet')
    df.write_parquet(final_path)
    return df

def cmp_std(x):
    li = x.split(',')
    li[0] = li[0][1:]
    li[-1] = li[-1][:-1]
    return np.std([float(x) for x in li])

def split(x):
    li = x.split(',')
    li[0] = li[0][1:]
    li[-1] = li[-1][:-1]
    return [float(x) for x in li]

def process_D135_136(df:pl.DataFrame, name:str):
    length = len(df[name][0].split(","))
    tdf = df[name].str.splitn(",",length).struct.rename_fields([name+'_'+str(i) for i in range(1, length+1)]).alias("fields").to_frame().unnest("fields")
    tdf = tdf.with_columns(pl.col(f"{name}_1").apply(lambda x: float(x[1:])))
    tdf = tdf.with_columns(pl.col(f"{name}_{length}").apply(lambda x: float(x[:-1])))
    for i in range(2,length):
        tdf = tdf.with_columns(pl.col(f"{name}_{i}").cast(pl.Float32))
    tdf = tdf.with_columns(tdf.mean(axis=1).alias(f'{name}_mean'))
    tdf = tdf.with_columns(tdf.min(axis=1).alias(f'{name}_min'))
    tdf = tdf.with_columns(tdf.max(axis=1).alias(f'{name}_max'))
    tdf = tdf.with_columns(df[name].apply(cmp_std).alias(f'{name}_std'))
    tdf = tdf.with_columns((tdf[f'{name}_max'] - tdf[f'{name}_min']).alias(f'{name}_diff'))
    return tdf

def process_charge_data(car_no, raw_data_path, output):
    dfs={}
    print(raw_data_path)
    vdcm_files, ptcan_files, bcan_files, bdcm_files = get_all_files(car_no, raw_data_path)
    ptcan_cols=['samplingtimesec','D19','D22','D24','D40','D41','D44','D27','D72','D52','D117','D133']
    bcan_cols=['samplingtimesec','D135', 'D140','D141','D136']
    vdcm_cols = ['samplingtimesec','D334', 'D356', 'D290']
    dfs['ptcan'] = concat_data(ptcan_files, ptcan_cols)
    dfs['ptcan'] = dfs['ptcan'].filter(pl.col('D19').is_between(2,5))
    dfs['bcan'] = concat_data(bcan_files, bcan_cols)
    dfs['bcan'] = dfs['bcan'].drop_nulls(subset=['D140','D141'])
    dfs['vdcm'] = concat_data(vdcm_files, vdcm_cols)
    df = dfs['bcan'].join(dfs['ptcan'], on='samplingtimesec', how='inner')
    del dfs['bcan'],dfs['ptcan']
    gc.collect()
    df = df.join(dfs['vdcm'].select(vdcm_cols), on='samplingtimesec', how='left')
    del dfs["vdcm"]
    gc.collect()
    df = df.filter(pl.col('D24').is_between(20, 80))
    df = df.with_columns(pl.col("D141").apply(lambda x: re.search('(?<=:)\d+', x)[0] if type(x)!=int else x))
    df = df.with_columns(pl.col("D140").apply(lambda x: re.search('(?<=:)\d+', x)[0] if type(x)!=int else x))
    tdf_D135 = process_D135_136(df, "D135")
    tdf_D136 = process_D135_136(df, "D136")
    df = df.drop(['D135', 'D136'])
    df = pl.concat([df,tdf_D135, tdf_D136], how="horizontal")
    df = df.with_columns(pl.col("samplingtimesec").cast(pl.Int64))
    df = df.sort("samplingtimesec")
    print(f"{car_no} has {df.shape[0]} records")
    df.filter(pl.col("D19")==3).write_parquet(output+f"{car_no}_3.parquet")
    df.filter((pl.col("D19")==4) | (pl.col("D19")==5)).write_parquet(output+f"{car_no}_4_5.parquet")
    return

def get_cycle_charge(df):
    CNT = 1
    def process_cycle(x):
        nonlocal CNT
        if x<-10:
            CNT+=1
            return CNT
        else:
            return CNT
    df = df.with_columns(pl.col("change").apply(process_cycle).alias("cycle"))
    return df
def get_cycle_discharge(df):
    CNT = 1
    def process_cycle(x):
        nonlocal CNT
        if x>5:
            CNT+=1
            return CNT
        else:
            return CNT
    df = df.with_columns(pl.col("change").apply(process_cycle).alias("cycle"))
    return df
# def process_charge_data2(df_charge):
#     df_charge = df_charge.with_columns(change=pl.col("D24").diff())
#     df_charge = df_charge.filter(pl.col("change")<1)
#     df_charge = df_charge.sort("samplingtimesec")
#     df_charge = get_cycle_charge(df_charge)
def get_charge_data(car_no):
    df_charge = pl.read_parquet(f"data/{car_no}_4_5.parquet")
    df_charge = df_charge.with_columns(change=pl.col("D24").diff())
    df_charge = df_charge.filter(pl.col("change")<1)
    df_charge = df_charge.sort("samplingtimesec")
    df_charge = get_cycle_charge(df_charge)
    return df_charge
def get_discharge_data(car_no):
    df_discharge = pl.read_parquet(f"data/{car_no}_3.parquet")
    df_discharge = df_discharge.filter(pl.col('D22')>=0)
    df_discharge = df_discharge.sort("samplingtimesec")
    df_discharge = df_discharge.with_columns(change=pl.col("D24").diff())
    df_discharge = get_cycle_discharge(df_discharge)
    return df_discharge
def preprocess(df):
    df = df.with_columns(pl.col('D24').floor().alias("D24_int"))
    df = df.with_columns(soc_diff = pl.col('D40')-pl.col('D41'))
    df = df.with_columns([pl.col(f"D135_{i}").cast(pl.Float32) for i in range(1,113)])
    df = df.with_columns([pl.col(col_name).cast(pl.Int32) for col_name in ['cycle','D24_int','label']])
    df = df.with_columns([pl.col(f"D136_{i}").cast(pl.Float32) for i in range(1,29)])
    df = df.with_columns([pl.col(col_name).cast(pl.Float32) for col_name in ['D27','soc_diff','D22','D356','D52','D334','D290']])
    return df
def get_col_dict():
    col_dict={}
    col_dict['cycle']=pl.Int32
    col_dict['D24_int']=pl.Int32
    col_dict['label']=pl.Int32
    for col in [f"D135_{i}" for i in range(1,113)]+[f"D136_{i}" for i in range(1,28)]+['D27','soc_diff','D22','D356','D52','D334','D290']:
        col_dict[col] = pl.Float32
    return col_dict

def process_data(df):
    if df.shape[0]>=seq_len:
        return df.sample(seq_len).sort("samplingtimesec").select(cols)
    else:
        tmp = df.select(cols)
        tmp = tmp.to_numpy()
        n = len(tmp)
        tmp = tmp.transpose()
        res = np.dstack([np.pad(tmp[i], pad_width=(0,seq_len-n),mode="edge") for i in range(len(cols))])
        return pl.DataFrame(np.squeeze(res,axis=0),schema=col_dict)
    
def make_label(df_charge,target_dt=''):
    if target_dt != '':
        df_tmp = df_charge.groupby("cycle").agg(D24_range=(pl.col('D24').max()-pl.col('D24').min()),
                              startTime=pl.col('samplingtimesec').min(),
                              endTime=pl.col('samplingtimesec').max()).sort('D24_range',descending=True)
        target_dt = time.strptime(target_dt, "%Y-%m-%d")
        target_dt = time.mktime(target_dt)*1000
        sep_cycle = df_tmp.filter((target_dt - pl.col('endTime'))<=0)['cycle'].min()-10
        df_charge=df_charge.with_columns(pl.col("cycle").apply(lambda x: 1 if x>=sep_cycle else 0).alias("label"))
    else:
        df_charge=df_charge.select(pl.lit(0).alias('label'),pl.all())
    return df_charge

def process_data_sample(target_car_dict):
    for item in normal_car_no:
        target_car_dict[item]=''
    for car, target_dt in target_car_dict.items():
        print(car)
        df_charge = get_charge_data(car)
        df_charge = make_label(df_charge, target_dt)
        df_charge = preprocess(df_charge)
        res_data = df_charge.groupby(['cycle','D24_int']).apply(process_data)
        res_data.write_parquet(f"processed_data/{car}_4_5.parquet")
        del df_charge
        del res_data
        df_discharge = get_discharge_data(car)
        df_discharge = make_label(df_discharge, target_dt)
        df_discharge = preprocess(df_discharge)
        res_data = df_discharge.groupby(['cycle','D24_int']).apply(process_data)
        res_data.write_parquet(f"processed_data/{car}_3.parquet")
    return
# if __name__ == '__main__':
    