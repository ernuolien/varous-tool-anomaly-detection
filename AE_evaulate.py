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
from normalizing_flow import NormalizingFlow, get_loss, get_loss_per_sample
from voraus_ad import ANOMALY_CATEGORIES, Signals, load_torch_dataloaders
# from CAE import ConvAutoencoder,Encoder,Decoder
from AE import AutoEncoder
import time
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

model = AutoEncoder().to(DEVICE)
criterion = torch.nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(),lr=configuration.learning_rate)
scheduler = optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=configuration.milestones, gamma=configuration.gamma
        )




model.load_state_dict(torch.load("AE_model32.pth"))

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
print(f"auroc(mean)={auroc_mean:5.3f}")


print("預測花費的時間：", prediction_time, "秒")
# print(aurocs)
# for auroc in aurocs:
#     print(auroc)



precision, recall, thresholds = metrics.precision_recall_curve(results["anomaly"], results["score"].values, pos_label=True)
f1_scores = 2 * (precision * recall) / (precision + recall)
optimal_idx = numpy.argmax(f1_scores)
optimal_threshold = thresholds[optimal_idx]

print("Optimal Threshold:", optimal_threshold)
# 计算 auroc
# 计算基于最佳阈值的预测标签
predicted_labels = (numpy.array(results["score"]) >= optimal_threshold).astype(int)

# 计算并打印准确率、召回率、精确率和 F1 分数
accuracy = numpy.mean(predicted_labels == results["anomaly"])
recall = recall[optimal_idx]
precision = precision[optimal_idx]
f1 = f1_scores[optimal_idx]

print(f'Accuracy: {accuracy}')
print(f'Recall: {recall}')
print(f'Precision: {precision}')
print(f'F1 Score: {f1}')
# 创建热图
from sklearn.metrics import accuracy_score
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
from sklearn.metrics import f1_score
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

normal_anomaly_scores = results[results['anomaly'] == False]['score']
threshold = numpy.percentile(normal_anomaly_scores, 70)
# threshold = 10000
print("Threshold:", threshold)
actual_labels = results["anomaly"]
anomaly_scores = results["score"].values    
predicted_labels = [1 if score > threshold else 0 for score in anomaly_scores]

# 计算准确率
accuracy = accuracy_score(actual_labels, predicted_labels)
recall = recall_score(actual_labels, predicted_labels)
precision = precision_score(actual_labels, predicted_labels)
f1score = f1_score(actual_labels, predicted_labels)
print("准确率:", accuracy)
print("recall ", recall)
print("precision ", precision)
print("f1score ", f1score)

conf_matrix = confusion_matrix(actual_labels, predicted_labels)

# 创建热图
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=['Pred 0', 'Pred 1'], yticklabels=['True 0', 'True 1'])
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('Confusion Matrix')
plt.show()
print("t")
