
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
    start_time = time.time()
    with torch.no_grad():
        result_list: List[Dict] = []
        maxloss_list = []
        minloss_list = []
        for _, (tensors, labels) in enumerate(test_dl):
            # if labels["category"][0] == "COLLISION_FOAM":
            #     end_time = time.time()
            tensors = tensors.float().to(DEVICE)
            # print(labels.items())
            # print(labels)
            # Calculate forward and jacobian.
            
            latent_z, jacobian = model.forward(tensors.transpose(2, 1))
            # if labels["category"][0][0] != "A":
            end_time = time.time()
            # print(latent_z.shape,jacobian.shape)
            jacobian = torch.sum(jacobian, dim=tuple(range(1, jacobian.dim())))
            # Calculate the anomaly score per sample.
            loss_per_sample = get_loss_per_sample(latent_z, jacobian)
            loss_per_sample_list.append(loss_per_sample)
            maxloss_list.append(max(loss_per_sample))
            minloss_list.append(min(loss_per_sample))
            # print(loss_per_sample.shape[0])
            # print(loss_per_sample)
            # Append the anomaly score and the labels to the results list.
            for j in range(loss_per_sample.shape[0]):
                result_labels = {k: v[j].item() if isinstance(v, torch.Tensor) else v[j] for k, v in labels.items()}
                result_labels.update(score=loss_per_sample[j].item())
                result_list.append(result_labels)
            # print(result_list)
    # end_time = time.time()
    
    print("avg of the max: ",sum(maxloss_list)/len(maxloss_list))
    print("avg of the min: ",sum(minloss_list)/len(minloss_list))
    print(max(minloss_list))
    prediction_time = end_time - start_time
    results = pandas.DataFrame(result_list)
    # Calculate AUROC per anomaly category.
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
        print("Optimal Threshold:", optimal_threshold)
        
        # 计算 auroc
        auroc = metrics.auc(fpr, tpr)
        aurocs.append(auroc)
        print(f'{category.name}, auroc={auroc:5.3f},')
        
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
        # print(f'{category.name}, accuracy={accuracy:5.3f}, recall={recall:5.3f}, precision={precision:5.3f}, f1_score={f1:5.3f}')
    # for category in ANOMALY_CATEGORIES:
    #     dfn = results[(results["category"] == category.name) | (~results["anomaly"])]
    #     fpr, tpr, thresholds = metrics.roc_curve(dfn["anomaly"], dfn["score"].values, pos_label=True)
    #     optimal_idx = numpy.argmax(tpr - fpr)
    #     optimal_threshold = thresholds[optimal_idx]
    #     print("Optimal Threshold:", optimal_threshold)
    #     # fpr, tpr, _ = metrics.roc_curve(dfn["anomaly"], dfn["score"].values, pos_label=True)
    #     auroc = metrics.auc(fpr, tpr)
    #     aurocs.append(auroc)
    #     print(f'{category.name}, auroc={auroc:5.3f},')
    # Calculate the AUROC mean over all categories.
    aurocs_array = numpy.array(aurocs)
    auroc_mean = aurocs_array.mean()
    print(f'accuracy={numpy.mean(acc_list):5.3f}, recall={numpy.mean(recall):5.3f}, precision={numpy.mean(precision):5.3f}, f1_score={numpy.mean(f1):5.3f}')
    # training_results.append({"epoch": epoch, "aurocMean": auroc_mean, "loss": loss})
    print(f" auroc(mean)={auroc_mean:5.3f}")
    print("預測花費的時間：", prediction_time, "秒")
    # scheduler.step()
#     from sklearn.metrics import accuracy_score
#     from sklearn.metrics import recall_score
#     from sklearn.metrics import precision_score
#     from sklearn.metrics import f1_score


#     # 假设实际标签和预测的异常分数如下
#     actual_labels = results["anomaly"]
#     anomaly_scores = results["score"]

#     # 设定阈值
#     # threshold = 0.74222 f1score = 0.8912 recall = 0.9337 precision = 0.8524
#     # threshold = 0.737 f1score = 0.8845 recall = 0.9788 precision = 0.8067
#     threshold = -395000

#     # 将异常分数转换为预测标签
#     predicted_labels = [1 if score > threshold else 0 for score in anomaly_scores]

#     # 计算准确率
#     accuracy = accuracy_score(actual_labels, predicted_labels)
#     recall = recall_score(actual_labels, predicted_labels)
#     precision = precision_score(actual_labels, predicted_labels)
#     f1score = f1_score(actual_labels, predicted_labels)

#     print("准确率:", accuracy)
#     print("recall ", recall)
#     print("precision ", precision)
#     print("f1score ", f1score)
#     # return training_results
#     with open(FILE_PATH, 'w') as file:
#         for item in loss_per_sample_list:
#             file.write("%s\n" % item)

if __name__ == "__main__":
    train()
