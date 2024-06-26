"""Contains the training of the normalizing flow model."""

import random
from pathlib import Path
from typing import Dict, List, Optional

import numpy
import pandas
import torch
import torch.backends.cudnn
from sklearn import metrics
from torch import optim

from configuration import Configuration
from normalizing_flow import NormalizingFlow, get_loss, get_AXIS_loss_per_sample
from voraus_ad import ANOMALY_CATEGORIES, Signals, load_torch_dataloaders
# from CAE import ConvAutoencoder,Encoder,Decoder
from AE import AutoEncoder
from LSTM_VAE import LSTMAutoencoder
from CAE import ConvAutoencoder
from sklearn.ensemble import IsolationForest
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import numpy as np
import joblib
# If deterministic CUDA is activated, some calculations cannot be calculated in parallel on the GPU.
# The training will take much longer but is reproducible.
DETERMINISTIC_CUDA = False
DATASET_PATH = Path.home() / "Downloads" / "voraus-ad-dataset-100hz.parquet"
MODEL_PATH: Optional[Path] = Path.cwd() / "model.pth"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
configuration = Configuration(
    columns="machine",
    epochs=70,
    frequencyDivider=1,
    trainGain=1.0,
    seed=177,
    batchsize=32,
    nCouplingBlocks=4,
    clamp=1.2,
    learningRate=8e-4,
    normalize=True,
    pad=True,
    nHiddenLayers=0,
    scale=2,
    kernelSize1=13,
    dilation1=2,
    kernelSize2=1,
    dilation2=1,
    kernelSize3=1,
    dilation3=1,
    milestones=[11, 61],
    gamma=0.1,
)
# Make the training reproducible.
torch.manual_seed(configuration.seed)
torch.cuda.manual_seed_all(configuration.seed)
numpy.random.seed(configuration.seed)
random.seed(configuration.seed)
if DETERMINISTIC_CUDA:
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

train_dataset, _, train_dl, test_dl = load_torch_dataloaders(
        dataset=DATASET_PATH,
        batch_size=configuration.batchsize,
        columns=Signals.groups()[configuration.columns],
        seed=configuration.seed,
        frequency_divider=configuration.frequency_divider,
        train_gain=configuration.train_gain,
        normalize=configuration.normalize,
        pad=configuration.pad,
    )

#####################################################################################
iforest_model =  IsolationForest(n_estimators=400, contamination=0.01)  
model = AutoEncoder().to(DEVICE)
criterion = torch.nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(),lr=configuration.learning_rate)
scheduler = optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=configuration.milestones, gamma=configuration.gamma
        )

model.load_state_dict(torch.load("voraus-ad-dataset-main/AE_model8.pth"))
#####################################################################################
# iforest_model =  IsolationForest(n_estimators=100, contamination=0.6)  
# model = LSTMAutoencoder(seq_len=1164, n_features=130, embedding_dim=32).to(DEVICE)
# criterion = torch.nn.MSELoss()
# optimizer = torch.optim.Adam(model.parameters(),lr=configuration.learning_rate)
# scheduler = optim.lr_scheduler.MultiStepLR(
#         optimizer, milestones=configuration.milestones, gamma=configuration.gamma
#     )

# iforest_model =  IsolationForest(n_estimators=100, contamination=0.1)  
# model.load_state_dict(torch.load("voraus-ad-dataset-main/LSTM_VAEmodel.pth"))
#####################################################################################
# iforest_model =  IsolationForest(n_estimators=100, contamination=0.6)  
# model = ConvAutoencoder().to(DEVICE)
# criterion = torch.nn.MSELoss()
# optimizer = torch.optim.Adam(model.parameters(),lr=configuration.learning_rate)
# scheduler = optim.lr_scheduler.MultiStepLR(
#         optimizer, milestones=configuration.milestones, gamma=configuration.gamma
#     )
# model.load_state_dict(torch.load("voraus-ad-dataset-main\CAEmodel.pth"))

model.eval()
with torch.no_grad():
    latent_features = []
    for _, (tensors, labels) in enumerate(train_dl):
        inputs = tensors.float().to(DEVICE)  # 假設數據是(input, label)形式
        # print(inputs.shape)
        encoded = model.encoder(inputs)
        # print(encoded.shape)
    # 将批次中的样本逐个传递给模型
        for sample in encoded:
            # 将样本转换为 NumPy 数组，并且根据需要进行形状调整
            sample_np = sample.cpu().detach().numpy().reshape(-1, 8)  # 假设有 130 个特征
            # 在 Isolation Forest 模型中进行训练
            iforest_model.fit(sample_np)

# 將潛在特徵轉換為NumPy數組
# 將潛在特徵數據展平

joblib.dump(iforest_model, 'AE_iforest_model400.pkl')