import numpy as np
import pandas as pd
import logging
import pickle
import glob
import os
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import LabelEncoder


import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torchaudio

original_sample_rate = 44100
target_sample_rate = 22050
channel = 0


n_mfcc = 40
log_mels = False


batch_size = 16
num_epochs = 10
num_classes = 77
clip_length = 1109
# learning_rate = 0.8
dropout_rate = 0.2
d_model = 13 # feature dimention
dim_feedforward = 2048 # number of hidden
num_layers = 2 # number of layers
nhead = 13 # number of heads
activation_function = 'relu'
clip = 5


cpu_count = os.cpu_count()


with open('label_encoder.pickle', 'rb') as input_file:
    label_encoder = pickle.load(input_file)

sample_transform = torchaudio.transforms.Resample(orig_freq=original_sample_rate, new_freq=target_sample_rate)
mfcc_transform = torchaudio.transforms.MFCC(sample_rate=target_sample_rate, n_mfcc=n_mfcc, log_mels=log_mels)

transform = nn.Sequential(sample_transform, mfcc_transform)


class Model(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward, dropout_rate, activation_function, num_layers):
        super(Model, self).__init__()
        
        encoder_layer = nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward=2048, dropout=dropout_rate, activation=activation_function)
        
        normalization = nn.LayerNorm(d_model)
                
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers, norm=normalization)
        
        self.batch_normalization = nn.BatchNorm1d(num_features=clip_length)

        self.dropout = nn.Dropout(p=dropout_rate)
        
        self.linear = nn.Linear(clip_length*d_model, num_classes)
        
        
    def forward(self, features):

        
        features = features.permute(2, 0, 1) # from (N,E,S) to (S,N,E)
        
        encoder_output = self.transformer_encoder(features)
        
        encoder_output = encoder_output.permute(1, 0, 2) # from (S,N,E) to (N,S,E)
        
        output_normalized = self.batch_normalization(encoder_output)
        
        output_dropped = self.dropout(output_normalized)
      
        output_reshaped = output_dropped.reshape(-1, clip_length*d_model) # from (N,S,E) to (N,S*E)
                        
        linear_output = self.linear(output_reshaped)
                                                                
        return linear_output


model = Model(d_model, nhead, dim_feedforward, dropout_rate, activation_function, num_layers)
model.load_state_dict(torch.load('model_state_dict'))

def predict(path):
    waveform, sample_rate = torchaudio.load(path)
    sound_data = transform(waveform[channel, :])[0:13, :].unsqueeze(0)

    model.eval()
    
    with torch.no_grad():
        output = model(sound_data)

    qari = label_encoder.inverse_transform([torch.argmax(output).item()])[0]
        
    return qari

