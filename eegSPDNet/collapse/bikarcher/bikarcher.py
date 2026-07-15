# Lib
import sys
from xml.parsers.expat import model
sys.path.append("/home/rffl/files/ic/utils/")
import torch
import datagen, eegdataset #type: ignore
from spd_learn.modules import Shrinkage, SPDBatchNormMeanVar
from spd_learn.models import EEGSPDNet
from torch.utils.data import DataLoader
from torchmetrics.classification import ConfusionMatrix
import matplotlib.pyplot as plt
import numpy as np
import pyriemann as rmn
import time

class EEGSPDNetLWF(EEGSPDNet):
    def __init__(self, *args, initShrinkage=0.0, **kwargs):
        super().__init__(*args, **kwargs)
        n_cov = self.n_chans * self.n_filters
        self.shrinkage = Shrinkage(n_chans=n_cov, init_shrinkage=initShrinkage, learnable=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.cov_pool(x)
        x = self.shrinkage(x)
        x = self.spdnet(x)
        x = self.dropout(x)
        x = self.linear(x)
        return x
    
class TSEEGNet(EEGSPDNetLWF):
    def __init__(self, *args, initShrinkage=0.0, spdbn_momentum=0.1, karcher=1, **kwargs):
        super().__init__(*args, initShrinkage=initShrinkage, **kwargs)
        k, n_bimap = self.bimap_sizes
        initial_size = self.n_chans * self.n_filters
        final_size = int((self.n_chans * self.n_filters) / k ** n_bimap)
        self.spdbn_prespd = SPDBatchNormMeanVar(
            num_features=initial_size,
            momentum=spdbn_momentum,
            bias_requires_grad=False,  # G-phi fixed to I
            weight_requires_grad=True,
            n_iter=karcher
        )
        self.spdbn_postspd = SPDBatchNormMeanVar(
            num_features=final_size,
            momentum=spdbn_momentum,
            bias_requires_grad=False,  # G-phi fixed to I
            weight_requires_grad=True,
            n_iter=karcher
        )

    def forward(self, x):
        x = self.conv(x)
        x = self.cov_pool(x)
        x = self.shrinkage(x)
        x = self.spdbn_prespd(x)
        x = self.spdnet[:-1](x)   # everything up to (but not including) LogEig
        x = self.spdbn_postspd(x)
        x = self.spdnet[-1](x)    # LogEig
        x = self.dropout(x)
        x = self.linear(x)
        return x

def regular_train(model, classVec, trainLoader, testLoader, epochs, device):
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.5e-3)

    train_hist_loss = []
    test_hist_loss = []

    train_hist_acc = []
    test_hist_acc = []

    t = time.time()
    print(f"Begin {t}")
    for epoch in range(epochs):
        train_loss, train_acc = train(model, trainLoader, criterion, optimizer, device)
        test_loss, test_acc = test(model, testLoader, criterion, device)
        train_hist_loss.append(train_loss); train_hist_acc.append(train_acc)
        test_hist_loss.append(test_loss); test_hist_acc.append(test_acc)
        print(f"Epoch {epoch+1}: "
        f"Train loss {train_loss:.4f}, acc {train_acc:.4f} | "
        f"Test loss {test_loss:.4f}, acc {test_acc:.4f} | "
        f"Time {time.time() - t} s")
        t = time.time()

    confmat = ConfusionMatrix(task="multiclass", num_classes=len(classVec))    

    fig, axs = plt.subplots(1,2)
    axs[0].plot(range(epochs), train_hist_loss, label="Train Loss")
    axs[0].plot(range(epochs), test_hist_loss, label="Test Loss")
    axs[0].legend()

    axs[1].plot(range(epochs), train_hist_acc, label="Train Accuracy")
    axs[1].plot(range(epochs), test_hist_acc, label="Test Accuracy")
    axs[1].legend()
    fig.savefig("train_test_loss_acc.png")

    for X, y in testLoader:
        X = X.to(device)
        y = y.to(device)

        logits = model(X)
        preds = torch.argmax(logits, dim=1)

        confmat.update(preds.cpu(), y.cpu())

    confmatrix = confmat.compute()
    plot_confusion_matrix(confmatrix, classVec)

def train(model, trainLoader, criterion, optimizer, device):
    torch.cuda.empty_cache()
    model.train()
    total_loss = 0
    correct = 0
    total = 0

    for X, y in trainLoader:
        X = X.to(device); y = y.to(device)

        optimizer.zero_grad()
        y_hat = model(X)
        loss = criterion(y_hat, y)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        pred = y_hat.argmax(dim=1)
        correct += pred.eq(y).sum().item()
        total += y.size(0)
    
    avg_loss = total_loss / len(trainLoader)
    acc = correct / total
    return avg_loss, acc

@torch.no_grad()
def test(model, testLoader, criterion, device):
    torch.cuda.empty_cache()
    model.eval()
    total_loss = 0
    correct = 0
    total = 0

    for X, y in testLoader:
        X = X.to(device); y = y.to(device)

        y_hat = model(X)
        total_loss += criterion(y_hat, y).item()

        pred = y_hat.argmax(dim=1)
        correct += pred.eq(y).sum().item()
        total += y.size(0)
    
    avg_loss = total_loss / len(testLoader)
    acc = correct / total
    return avg_loss, acc

def plot_confusion_matrix(confmatrix, class_names=None):
    plt.figure(figsize=(6, 5))
    plt.imshow(confmatrix, interpolation='nearest')
    plt.title("Confusion Matrix")
    plt.colorbar()

    num_classes = confmatrix.shape[0]

    # Tick labels
    if not class_names:
        class_names = [str(i) for i in range(num_classes)]

    plt.xticks(range(num_classes), class_names, rotation=45)
    plt.yticks(range(num_classes), class_names)

    # Print values inside cells
    for i in range(num_classes):
        for j in range(num_classes):
            plt.text(j, i, str(confmatrix[i, j].item()),
                     ha="center", va="center", color="white" if confmatrix[i, j] > confmatrix.max()/2 else "black")

    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.tight_layout()
    plt.savefig("confusion_matrix.png")

def main():
    device = "cuda:1" if torch.cuda.is_available() else "cpu"
    device

    # Data config
    classVec = [i+1 for i in range(10)]
    subVec = [i+1 for i in range(10)]

    nClass = len(classVec)
    nSub = len(subVec)

    # Data gen
    train_path, test_path = datagen.generatedataset(classVec, subVec, 0.75, False)

    # xDAWN treatment
    xDAWNfilters = 5
    estimator = "lwf"

    X_train = np.load(train_path, allow_pickle=True)
    X_test = np.load(test_path, allow_pickle=True)

    y_train = np.array([])
    part = len(X_train) // len(classVec)
    for i in range(len(classVec)):
        y_train = np.concatenate([y_train, i * np.ones(part)])

    xdawn = rmn.estimation.Xdawn(nfilter=xDAWNfilters, estimator=estimator)
    xdawn.fit(X_train, y_train)

    X_train = np.einsum('fc,nct->nft', xdawn.filters_, X_train)
    X_test  = np.einsum('fc,nct->nft', xdawn.filters_, X_test)

    np.save(train_path, X_train )
    np.save(test_path, X_test )

    # Dataset and Dataloader init
    trainDataset = eegdataset.KClassDataset(nClass, "./train.npy")
    trainLoader = DataLoader(trainDataset, 40, True, num_workers=4)

    testDataset = eegdataset.KClassDataset(nClass, "./test.npy")
    testLoader = DataLoader(testDataset, 40, True, num_workers=4)

    device = torch.device(device)
    model = TSEEGNet(50, 10, n_filters=10, bimap_sizes=(2, 1), filter_time_length = 50, final_layer_drop_prob=0.25, spd_drop_prob=0, karcher=3).to(device)
    regular_train(model, classVec, trainLoader, testLoader, 75, device)

main()