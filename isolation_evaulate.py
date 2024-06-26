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
import math,time

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

def transform_number(num):
    if num > 1:
        return math.log(num)  # 將大數字變小
    else:
        return math.exp(num)  # 將小數字變大
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
# Autoencoder + isolation forest
model = AutoEncoder().to(DEVICE)
criterion = torch.nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(),lr=configuration.learning_rate)
scheduler = optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=configuration.milestones, gamma=configuration.gamma
        )
model.load_state_dict(torch.load("voraus-ad-dataset-main/AE_model8.pth"))

iforest_model = joblib.load('voraus-ad-dataset-main/AE_iforest_model30.pkl')

#####################################################################################
#LSTMAutoencoder + isolation forest
# model = LSTMAutoencoder(seq_len=1164, n_features=130, embedding_dim=32).to(DEVICE)
# criterion = torch.nn.MSELoss()
# optimizer = torch.optim.Adam(model.parameters(),lr=configuration.learning_rate)
# scheduler = optim.lr_scheduler.MultiStepLR(
#         optimizer, milestones=configuration.milestones, gamma=configuration.gamma
#         )
# model.load_state_dict(torch.load("voraus-ad-dataset-main/LSTM_VAEmodel.pth"))

# iforest_model = joblib.load('voraus-ad-dataset-main/LSTM_VAE_iforest_model.pkl')

#####################################################################################

total_loss = 0
start_time = time.time()
model.eval()
with torch.no_grad():
    result_list = []
    for _, (tensors, labels) in enumerate(test_dl):
        inputs = tensors.float().to(DEVICE)
        
        encoded = model.encoder(inputs)
        # outputs = model(inputs)
        anomaly_scores = []
        # loss_per_sample = get_loss_per_sample(inputs, outputs)
        # print(loss_per_sample)
        for sample in encoded:
            sample_np = sample.cpu().numpy().reshape(-1, 8)
            anomalyscores_perseconds = iforest_model.decision_function(sample_np)
            anomaly_scores_persample = sum(anomalyscores_perseconds)/len(anomalyscores_perseconds)
            anomaly_scores.append(anomaly_scores_persample)
        
        # print(anomaly_scores)
        for j in range(len(anomaly_scores)):
            result_labels = {k: v[j].item() if isinstance(v, torch.Tensor) else v[j] for k, v in labels.items()}
            result_labels.update(score=anomaly_scores[j]) #.item()
            result_list.append(result_labels)
    # print(f'Test Loss: {total_loss / len(test_dl):.4f}')
    # print(result_list["score"])
end_time = time.time()
prediction_time = end_time - start_time
results = pandas.DataFrame(result_list)
# Calculate AUROC per anomaly category.
aurocs = []
for category in ANOMALY_CATEGORIES:
    dfn = results[(results["category"] == category.name) | (~results["anomaly"])]
    fpr, tpr, _ = metrics.roc_curve(dfn["anomaly"], dfn["score"].values, pos_label=True)
    auroc = metrics.auc(fpr, tpr)
    aurocs.append(auroc)
    print(f'{category.name}, auroc={auroc:5.3f},')
# Calculate the AUROC mean over all categories.
aurocs_array = numpy.array(aurocs)
auroc_mean = aurocs_array.mean()
print(f"auroc(mean)={auroc_mean:5.3f}")
print("預測花費的時間：", prediction_time, "秒")