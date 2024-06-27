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
import time
from configuration import Configuration
from normalizing_flow import NormalizingFlow, get_loss, get_loss_per_sample
from voraus_ad import ANOMALY_CATEGORIES, Signals, load_torch_dataloaders
from CAEwithAtt import ConvAutoencoder
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

model = ConvAutoencoder().to(DEVICE)
criterion = torch.nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(),lr=configuration.learning_rate)
scheduler = optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=configuration.milestones, gamma=configuration.gamma
        )




model.load_state_dict(torch.load("CAEwithSESAmodel870.pth"))

total_loss = 0
model.eval()
start_time = time.time()
with torch.no_grad():
    result_list = []
    for _, (tensors, labels) in enumerate(test_dl):
        
        inputs = tensors.float().to(DEVICE)
        # print(labels.shape)
        
        outputs = model(inputs)
        # loss = criterion(outputs, inputs)
        
        # print(loss)
        # total_loss += loss.item()
        loss_per_sample = get_loss_per_sample(inputs, outputs)
        # print(loss_per_sample)
        for j in range(loss_per_sample.shape[0]):
            result_labels = {k: v[j].item() if isinstance(v, torch.Tensor) else v[j] for k, v in labels.items()}
            result_labels.update(score=loss_per_sample[j].item())
            result_list.append(result_labels)

    print(f'Test Loss: {total_loss / len(test_dl):.4f}')
# print(result_list)
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
# print(results)
print(f"auroc(mean)={auroc_mean:5.3f}")
print("單次預測花費的時間：", prediction_time, "秒")



