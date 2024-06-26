
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
# from normalizing_flow import NormalizingFlow, get_loss
from normalizing_flow_att import NormalizingFlow
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
    model = NormalizingFlow((n_signals, n_times), configuration).float()
    optimizer = torch.optim.Adam(model.parameters(), lr=configuration.learning_rate)
    scheduler = optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=configuration.milestones, gamma=configuration.gamma
    )

    model.load_state_dict(torch.load("voraus-ad-dataset-main/ngraph_attmodel.pth"))

    training_results: List[Dict] = []
    # for epoch in range(configuration.epochs):
    #     loss: float = 0
    model.eval()
    start_time = time.time()
    with torch.no_grad():
        result_list: List[Dict] = []
        for _, (tensors, labels) in enumerate(test_dl):
            # if labels["category"][0] == "COLLISION_FOAM":
            #     end_time = time.time()
            tensors = tensors.float()
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
            # print(loss_per_sample.shape[0])
            # print(loss_per_sample)
            # Append the anomaly score and the labels to the results list.
            for j in range(loss_per_sample.shape[0]):
                result_labels = {k: v[j].item() if isinstance(v, torch.Tensor) else v[j] for k, v in labels.items()}
                result_labels.update(score=loss_per_sample[j].item())
                result_list.append(result_labels)
            # print(result_list)
    # end_time = time.time()
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
    # training_results.append({"epoch": epoch, "aurocMean": auroc_mean, "loss": loss})
    print(f" auroc(mean)={auroc_mean:5.3f}")
    print("預測花費的時間：", prediction_time, "秒")
    # scheduler.step()

    # return training_results
    with open(FILE_PATH, 'w') as file:
        for item in loss_per_sample_list:
            file.write("%s\n" % item)

if __name__ == "__main__":
    train()
