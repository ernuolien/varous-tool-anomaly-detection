"""Contains the training of the normalizing flow model."""

import random
from pathlib import Path
from typing import Dict, List, Optional

import numpy
import pandas
import torch
from torch import Tensor, nn
import torch.backends.cudnn
from sklearn import metrics
from torch import optim
# from normalizing_flow import  get_loss, get_loss_per_sample
from configuration import Configuration
# from normalizing_flow import get_loss_per_sample, get_loss
from voraus_ad import ANOMALY_CATEGORIES, Signals, load_torch_dataloaders
# from CAE import ConvAutoencoder,Encoder,Decoder
from AE import AutoEncoder
from AEwithAtt import AutoEncoderWithAttention,SelfAttention
# If deterministic CUDA is activated, some calculations cannot be calculated in parallel on the GPU.
# The training will take much longer but is reproducible.
DETERMINISTIC_CUDA = False
DATASET_PATH = Path.home() / "Downloads" / "voraus-ad-dataset-100hz.parquet"
MODEL_PATH: Optional[Path] = Path.cwd() / "model.pth"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", DEVICE)
def get_loss(inputs: Tensor, outputs: Tensor) -> Tensor:
    """Calculate the loss of a batch.

    Returns:
        The loss of the batch.
    """
   
    number = numpy.prod(inputs.shape[1:])
    sum_dimension = tuple(range(1, inputs.dim()))
    return torch.mean(torch.sum((inputs-outputs)**2, dim=sum_dimension)) / number
def get_loss_per_sample(inputs: Tensor, outputs: Tensor) -> Tensor:
    """Calculates the loss per sample.

    Args:
        z_space: The batch result.
        jac: The jacobian matrix.

    Returns:
        The loss per sample.
    """
    sum_dimension = tuple(range(1, inputs.dim()))
    # loss = 0.5 * torch.sum(z_space**2, dim=sum_dimension) - jac
    loss = 0.5 * torch.sum((inputs-outputs)**2, dim=sum_dimension)
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
    learningRate=1e-3,
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
    
    # 使用设定的参数创建 CVAE 实例
    # model = CVAE(in_channels, latent_dim).to(DEVICE)
    # attention = SelfAttention(130,32).to(DEVICE)
    model = AutoEncoderWithAttention().to(DEVICE)
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(),lr=configuration.learning_rate)
    scheduler = optim.lr_scheduler.MultiStepLR(
            optimizer, milestones=configuration.milestones, gamma=configuration.gamma
        )
    training_results: List[Dict] = []
    # 舊的basic CAE
    for epoch in range(configuration.epochs):
        for data in train_dl:
            inputs, _ = data
            inputs = inputs.float().to(DEVICE)
            # print(inputs.shape)
            optimizer.zero_grad()
            outputs = model(inputs)
            # inputs_attention = attention(inputs)
            # loss = criterion(outputs, inputs)
            loss = get_loss(inputs, outputs)
            loss.backward()
            optimizer.step()
            
    
        print(f'Epoch [{epoch+1}/{configuration.epochs}], Loss: {loss.item():.4f}')
        if MODEL_PATH is not None:
            torch.save(model.state_dict(), MODEL_PATH)

        total_loss = 0
        
        model.eval()
        with torch.no_grad():
            result_list = []
            for _, (tensors, labels) in enumerate(test_dl):
                inputs = tensors.float().to(DEVICE)
                # print(labels.shape)
                outputs = model(inputs)
                # loss = criterion(outputs, inputs)

                # print(loss)
                # total_loss += loss.item()
                # inputs_attention = attention(inputs)
                loss_per_sample = get_loss_per_sample(inputs, outputs)
                # print(loss_per_sample)
                for j in range(loss_per_sample.shape[0]):
                    result_labels = {k: v[j].item() if isinstance(v, torch.Tensor) else v[j] for k, v in labels.items()}
                    result_labels.update(score=loss_per_sample[j].item())
                    result_list.append(result_labels)
        
        results = pandas.DataFrame(result_list)

        # Calculate AUROC per anomaly category.
        aurocs = []
        for category in ANOMALY_CATEGORIES:
            dfn = results[(results["category"] == category.name) | (~results["anomaly"])]
            fpr, tpr, _ = metrics.roc_curve(dfn["anomaly"], dfn["score"].values, pos_label=True)
            auroc = metrics.auc(fpr, tpr)
            aurocs.append(auroc)
            # print(f'{category.name}, auroc={auroc:5.3f},')
        # Calculate the AUROC mean over all categories.
        aurocs_array = numpy.array(aurocs)
        auroc_mean = aurocs_array.mean()
        # print(results)
        training_results.append({"epoch": epoch, "aurocMean": auroc_mean, "loss": loss})
        print(f"Epoch {epoch:0>3d}: auroc(mean)={auroc_mean:5.3f}, loss={loss:.6f}")

        scheduler.step()
    for category in ANOMALY_CATEGORIES:
            dfn = results[(results["category"] == category.name) | (~results["anomaly"])]
            fpr, tpr, _ = metrics.roc_curve(dfn["anomaly"], dfn["score"].values, pos_label=True)
            auroc = metrics.auc(fpr, tpr)
            aurocs.append(auroc)
            print(f'{category.name}, auroc={auroc:5.3f},')
    
    return training_results

if __name__ == "__main__":
    train()
