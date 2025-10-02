import torch
from torch import nn, optim
import torch.nn.functional as F

class EEGNet(nn.Module):
    def __init__(self,C,T,R,N,F1 = 4, D = 2, P = 0.5):
        super(EEGNet, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=F1,kernel_size=(1,int(R/2)), padding="same")
        self.batchnorm1 = nn.BatchNorm2d(F1)
        self.conv2 = nn.Conv2d(in_channels=F1, out_channels=D*F1, kernel_size=(C,1), padding="valid", groups=F1)
        self.batchnorm2 = nn.BatchNorm2d(D*F1)
        self.elu = nn.ELU()
        self.dropout = nn.Dropout2d(p=P)
        self.pool1 = nn.AvgPool2d(kernel_size=(1,int(R/32))) # maybe the kernel_size is (1,4), need to refactor the linear
        self.conv3_depth = nn.Conv2d(in_channels=D*F1, out_channels=D*F1, kernel_size=(1,16), padding="same", groups=D*F1)
        self.conv3_point = nn.Conv2d(in_channels=D*F1, out_channels=D*F1, kernel_size=(1,1))
        self.batchnorm3 = nn.BatchNorm2d(D*F1)
        self.pool2 = nn.AvgPool2d(kernel_size=(1,8))
        self.flatten = nn.Flatten()
        self.dense = nn.Linear(in_features=4*D*F1*T//R, out_features=N)
    
    def forward(self, inp):
        x = self.conv1(inp)
        x = self.batchnorm1(x)
        x = self.conv2(x)      # maxnorm = 1
        x = self.batchnorm2(x)
        x = self.elu(x)
        x = self.pool1(x)
        x = self.dropout(x)
        x = self.conv3_depth(x)
        x = self.conv3_point(x)
        x = self.batchnorm3(x)
        x = self.elu(x)
        x = self.pool2(x)
        x = self.dropout(x)
        x = self.flatten(x)
        x = self.dense(x)      # maxnorm = 0.25
        return x

def regular_train(C, T, R, N, train_loader, test_loader, epochs = 20, cross = False):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    p = 0.25 if cross else 0.5
    model = EEGNet(C,T,R,N,P=p)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    for epoch in range(epochs):
        train_loss, train_acc = train(model, train_loader, criterion, optimizer, device)
        test_loss, test_acc = test(model, test_loader, criterion, optimizer, device)
        print(f"Epoch {epoch+1}: "
        f"Train loss {train_loss:.4f}, acc {train_acc:.4f} | "
        f"Test loss {test_loss:.4f}, acc {test_acc:.4f}")

def apply_maxnorm(model, maxnorm_conv=1.0, maxnorm_dense=0.25):
    with torch.no_grad():
        w = model.conv2.weight
        norms = w.view(w.size(0), -1).norm(2, dim=1, keepdim=True)
        desired = torch.clamp(norms, max=maxnorm_conv)
        w.view(w.size(0), -1).mul_(desired / (1e-8 + norms))

        w = model.dense.weight
        norms = w.norm(2, dim=1, keepdim=True)
        desired = torch.clamp(norms, max=maxnorm_dense)
        w.mul_(desired / (1e-8 + norms))

def train(model, train_loader, criterion, optimizer, device):
    model.train()
    total_loss = 0
    correct = 0
    total = 0

    for X, y in train_loader:
        X = X.to(device); y = y.to(device)

        optimizer.zero_grad()
        y_hat = model(X)
        loss = criterion(y_hat, y).item()
        loss.backward()
        optimizer.step()
        apply_maxnorm(model)

        total_loss += loss.item()
        pred = y.argmax(dim=1)
        correct += pred.eq(y).sum().item()
        total += y.size(0)
    
    avg_loss = total_loss / len(train_loader)
    acc = correct / total
    return avg_loss, acc

@torch.no_grad()
def test(model, test_loader, criterion, device):
    model.eval()
    total_loss = 0
    correct = 0
    total = 0

    for X, y in test_loader:
        X = X.to(device); y = y.to(device)

        y_hat = model(X)
        total_loss += criterion(y_hat, y).item()

        pred = y_hat.argmax(dim=1)
        correct += pred.eq(y).sum().item()
        total += y.size(0)
    
    avg_loss = total_loss / len(test_loader)
    acc = correct / total
    return avg_loss, acc