
import random
from pathlib import Path
from typing import Dict, List, Optional

import numpy
import pandas
import torch
import torch.backends.cudnn
from torch import Tensor
from sklearn import metrics
from torch import optim
import time
from sklearn.metrics import accuracy_score
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
from sklearn.metrics import f1_score
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from configuration import Configuration
from normalizing_flow import NormalizingFlow, get_loss
from voraus_ad import ANOMALY_CATEGORIES, Signals, load_torch_dataloaders

# If deterministic CUDA is activated, some calculations cannot be calculated in parallel on the GPU.
# The training will take much longer but is reproducible.
DETERMINISTIC_CUDA = False
DATASET_PATH = Path.home() / "Downloads" / "voraus-ad-dataset-100hz.parquet"
MODEL_PATH: Optional[Path] = Path.cwd() / "model.pth"
FILE_PATH: Optional[Path] = Path.cwd() / "loss.txt"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(DEVICE)
loss_per_sample_list = []
def get_loss_per_sample(z_space: Tensor, jac: Tensor) -> Tensor:
    """Calculates the loss per sample.

    Args:
        z_space: The batch result.
        jac: The jacobian matrix.

    Returns:
        The loss per sample.
    """
    sum_dimension = tuple(range(1, z_space.dim()))
    loss = 0.5 * torch.sum(z_space**2, dim=sum_dimension) - jac
    
    return loss
# Define the training configuration and hyperparameters of the model.
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


# Disable pylint too-many-variables here for readability.
# The whole training should run in a single function call.
def train() -> List[Dict]:  # pylint: disable=too-many-locals
    """Trains the model with the paper-given parameters.

    Returns:
        The auroc (mean over categories) and loss per epoch.
    """
    # Load the dataset as torch data loaders.
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

    # Retrieve the shape of the data for the model initialization.
    n_signals = train_dataset.tensors[0].shape[1]
    n_times = train_dataset.tensors[0].shape[0]
    # Initialize the model, optimizer and scheduler.
    model = NormalizingFlow((n_signals, n_times), configuration).float().to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=configuration.learning_rate)
    scheduler = optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=configuration.milestones, gamma=configuration.gamma
    )

    model.load_state_dict(torch.load("Normal_model.pth"))

    training_results: List[Dict] = []
    # for epoch in range(configuration.epochs):
    #     loss: float = 0
    model.eval()
    with torch.no_grad():
        result_list: List[Dict] = []
        start_time = time.time()
        for _, (tensors, labels) in enumerate(test_dl):
            tensors = tensors.float().to(DEVICE)
            # Calculate forward and jacobian.
            latent_z, jacobian = model.forward(tensors.transpose(2, 1))
            jacobian = torch.sum(jacobian, dim=tuple(range(1, jacobian.dim())))
            # Calculate the anomaly score per sample.
            loss_per_sample = get_loss_per_sample(latent_z, jacobian)
            # Append the anomaly score and the labels to the results list.
            for j in range(loss_per_sample.shape[0]):
                result_labels = {k: v[j].item() if isinstance(v, torch.Tensor) else v[j] for k, v in labels.items()}
                result_labels.update(score=loss_per_sample[j].item())
                result_list.append(result_labels)
            # print(result_list)
    # end_time = time.time()
    end_time = time.time()
    prediction_time = end_time - start_time
    print("單次預測花費的時間：", prediction_time, "秒")
    results = pandas.DataFrame(result_list)
    # Calculate AUROC per anomaly category.
    
    
#區分不同異常的最佳threshold計算各項指標(超爛)
    aurocs = []
    acc_list = []
    re_list = []
    pre_list = []
    f1_list = []
    for category in ANOMALY_CATEGORIES:
        dfn = results[(results["category"] == category.name) | (~results["anomaly"])]
        fpr, tpr, thresholds = metrics.roc_curve(dfn["anomaly"], dfn["score"].values, pos_label=True)
        optimal_idx = numpy.argmax(tpr - fpr)
        optimal_threshold = thresholds[optimal_idx]
        # print("Optimal Threshold:", optimal_threshold)
        
        # 计算 auroc
        auroc = metrics.auc(fpr, tpr)
        aurocs.append(auroc)
        # print(f'{category.name}, auroc={auroc:5.3f},')
        
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


# 計算正常數據中的分數（score）的第75百分位數，並將其用作threshold。
    normal_anomaly_scores = results[results['anomaly'] == False]['score']
    threshold = numpy.percentile(normal_anomaly_scores, 75)
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
    sns.set(font_scale=1.7)
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=['Pred Normal', 'Pred Anomaly'], yticklabels=['True Normal', 'True Anomaly'])
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title('Confusion Matrix')
    plt.show()
    print("t")


if __name__ == "__main__":
    train()
