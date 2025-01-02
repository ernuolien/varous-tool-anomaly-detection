"""Contains the training of the normalizing flow model."""

import random
from pathlib import Path
from typing import Dict, List, Optional

import numpy
import numpy as np
import pandas
import torch
import torch.backends.cudnn
from sklearn import metrics
from sklearn.metrics import accuracy_score
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
from sklearn.metrics import f1_score
from torch import optim
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve
import time
from configuration import Configuration
from normalizing_flow import NormalizingFlow, get_loss, get_loss_per_sample
from voraus_ad import ANOMALY_CATEGORIES, Signals, load_torch_dataloaders
# from CAE import ConvAutoencoder,Encoder,Decoder
from CAE import Encoder, Decoder, ConvAutoencoder
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




model.load_state_dict(torch.load("CAE_model3.pth"))

total_loss = 0
model.eval()
start_time = time.time()
with torch.no_grad():
    result_list = []
    for _, (tensors, labels) in enumerate(test_dl):
        inputs = tensors.float().to(DEVICE)
        outputs = model(inputs)
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


#找出最佳threshold
precision, recall, thresholds = metrics.precision_recall_curve(results["anomaly"], results["score"].values, pos_label=True)
f1_scores = 2 * (precision * recall) / (precision + recall)
optimal_idx = numpy.argmax(f1_scores)
optimal_threshold = thresholds[optimal_idx]
print("Optimal Threshold:", optimal_threshold)

#不分異常種類 使用最佳threshold判定所有資料是否異常
precision, recall, thresholds = precision_recall_curve(results["anomaly"], results["score"].values, pos_label=True)
predicted_labels = (np.array(results["score"]) >= optimal_threshold).astype(int)

accuracy = np.mean(predicted_labels == results["anomaly"])
recall = recall[optimal_idx]
precision = precision[optimal_idx]
f1 = f1_scores[optimal_idx]

print(f'Accuracy: {accuracy:.4f}')
print(f'Recall: {recall:.4f}')
print(f'Precision: {precision:.4f}')
print(f'最佳F1 Score: {f1:.4f}')

# 計算正常數據中的分數（score）的第65百分位數，並將其用作threshold。
actual_labels = results["anomaly"]
anomaly_scores = results["score"].values
normal_anomaly_scores = results[results['anomaly'] == False]['score']
threshold = numpy.percentile(normal_anomaly_scores, 65)
predicted_labels = [1 if score > threshold else 0 for score in anomaly_scores]
accuracy = accuracy_score(actual_labels, predicted_labels)
recall = recall_score(actual_labels, predicted_labels)
precision = precision_score(actual_labels, predicted_labels)
f1score = f1_score(actual_labels, predicted_labels)

print("Accuracy:", accuracy)
print("recall ", recall)
print("precision ", precision)
print("最現實f1score ", f1score)
# predicted_labels = [1 if score > optimal_threshold else 0 for score in anomaly_scores]
conf_matrix = metrics.confusion_matrix(actual_labels, predicted_labels)

# 创建热图
sns.set(font_scale=1.7)
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=['Pred Normal', 'Pred Anomaly'], yticklabels=['True Normal', 'True Anomaly'])
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('Confusion Matrix')
plt.show()
print("t")

#區分不同異常的最佳threshold計算各項指標(超爛)

for category in ANOMALY_CATEGORIES:
    dfn = results[(results["category"] == category.name) | (~results["anomaly"])]
    fpr, tpr, thresholds = metrics.roc_curve(dfn["anomaly"], dfn["score"].values, pos_label=True)
    optimal_idx = np.argmax(tpr - fpr)
    optimal_threshold = thresholds[optimal_idx]
    # print("Optimal Threshold:", optimal_threshold)
    
    # 计算 auroc
    auroc = metrics.auc(fpr, tpr)
    aurocs.append(auroc)
    # print(f'{category.name}, auroc={auroc:5.3f},')
    aurocs = []
    acc_list = []
    re_list = []
    pre_list = []
    f1_list = []
    # 根据最佳阈值计算预测标签
    predicted_labels = (dfn["score"].values >= optimal_threshold).astype(int)
    true_labels = dfn["anomaly"].values.astype(int)
    
    # 计算准确率
    accuracy = metrics.accuracy_score(true_labels, predicted_labels)
    acc_list.append(accuracy)
    # 计算召回率
    recall = metrics.recall_score(true_labels, predicted_labels)
    re_list.append(recall)
    # 计算精确率
    precision = metrics.precision_score(true_labels, predicted_labels)
    pre_list.append(precision)
    # 计算 F1 分数
    f1 = metrics.f1_score(true_labels, predicted_labels)
    f1_list.append(f1)
    # 打印指标
    print(f'{category.name}, accuracy={accuracy:5.3f}, recall={recall:5.3f}, precision={precision:5.3f}, f1_score={f1:5.3f}')