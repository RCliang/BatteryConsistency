import torch
import math
import copy
import os
import time
import enum
import argparse
import polars as pl
import torch.nn as nn
from torch.nn import functional as F
from torch.optim import Adam
from torch.utils.tensorboard import SummaryWriter
from torch.hub import download_url_to_file
import seaborn as sns


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
    def __init__(self, model_dimension, number_of_heads, number_of_layers, dropout_probability, number_of_var,
                 hidden_dimension_list):
        super().__init__()
        # Periodic Embeddings
        # two different encoders and decoders
        EncoderLayer = nn.TransformerEncoderLayer(d_model=model_dimension,
                                                  nhead=number_of_heads, batch_first=True)
        self.encoder = nn.TransformerEncoder(EncoderLayer, num_layers=number_of_layers)
        self.fw_embedding = nn.Linear(number_of_var, model_dimension)
        self.bw_embedding = nn.Linear(number_of_var, model_dimension)
        self.pos_embedding = PositionalEncoding(model_dimension, dropout_probability)

        DecoderLayer = nn.TransformerDecoderLayer(d_model=model_dimension,
                                                  nhead=number_of_heads, batch_first=True)
        self.decoder = nn.TransformerDecoder(DecoderLayer, num_layers=number_of_layers)
        self.mlp = nn.ModuleList([nn.Sequential(nn.Linear(model_dimension,hidden_dimension_list[0]),nn.ReLU())]
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
    
    def decode(self, input, input_repr, bw_mask, src_mask):
        input_embeddings = self.bw_embedding(input)
        # print(input_embeddings.shape)
        input_embeddings = self.pos_embedding(input_embeddings)
        bw_repr = self.decoder(input_embeddings, input_repr, tgt_mask=bw_mask, tgt_key_padding_mask=src_mask)
        return bw_repr
        
    def forward(self, input, src_mask):
        input_repr = self.encode(input, src_mask)
        # bw_mask = self.get_bw_mask(input)
        # decode_repr = self.decode(input, input_repr, bw_mask, src_mask)
        # for m in self.mlp:
        #     decode_repr = m(decode_repr)
        for m in self.mlp:
            input_repr = m(input_repr)
        pred = F.log_softmax(input_repr,dim=1)
        return pred