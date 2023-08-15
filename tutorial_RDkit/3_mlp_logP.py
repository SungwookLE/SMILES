import numpy as np
import matplotlib.pyplot as plt
plt.switch_backend('agg')
from utils import read_ZINC
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
import torch.nn.functional as F


class Preprocessor:
    def __init__(self):
        self.fps_total, self.logP_total, self.tpsa_total = read_ZINC(60000)
        self.num_train = 4000
        self.num_validation = 10000
        self.num_test = 10000 
      
    def train_test_split(self):
        self.fps_train = self.fps_total[0:self.num_train]
        self.logP_train = self.logP_total[0:self.num_train]
        self.fps_validation = self.fps_total[self.num_train:(self.num_train+self.num_validation)]
        self.logP_validation = self.logP_total[self.num_train:(self.num_train+self.num_validation)]
        self.fps_test = self.fps_total[(self.num_train+self.num_validation):]
        self.logP_test = self.logP_total[(self.num_train+self.num_validation):]
        return self.fps_train, self.logP_train, self.fps_validation, self.logP_validation, self.fps_test, self.logP_test

class CustomDataset(Dataset):
    def __init__(self, npData, npLabel, is_test=False):
        self.npData = npData # fps
        self.npLabel = npLabel # logP
        self.is_test = is_test # train, valid / test

    def __getitem__(self, index):
        fp = self.npData[index]
        if not self.is_test: 
            label = self.npLabel[index]
            return torch.tensor(fp).float(), torch.tensor(label).float()
        else:
            return torch.tensor(fp).float() # feature
    
    def __len__(self):
        return len(self.npData)


class MLP(nn.Module):
    def __init__(self, input_size, out_size, hidden_size=512):
        super(MLP, self).__init__()
        self.hidden_size = hidden_size
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.out = nn.Linear(hidden_size, out_size)

    def forward(self,X):
        out = self.fc1(X)
        out = F.relu(out)
        out = self.fc2(out)
        out = F.tanh(out)
        out = self.out(out)
        return out

def train(model, optim, criterion, dataloader):
    size = len(dataloader.dataset)
    batchSize = len(dataloader)
    batch_size = size/batchSize

    totalLoss = 0
    for batch, (X, y) in enumerate(dataloader):
        X = X.to(device)
        y = y.to(device)

        pred = model(X)

        # backpropagation
        optim.zero_grad()
        loss = criterion(pred, y)
        loss.backward()
        optim.step()
        totalLoss += loss.item()

        if (batch % 20 == 0):
            print(f"Loss: {loss.item():.3f}, {batch * batch_size}/{size}")


    meanLossPerBatch = totalLoss / batchSize
    return meanLossPerBatch

def test(model, criterion, dataloader):
    size = len(dataloader.dataset)
    batchSize = len(dataloader)
    batch_size = size/batchSize

    totalLoss = 0
    model.eval()
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)

        with torch.no_grad():
            pred = model(X)
            loss = criterion(pred, y)
            totalLoss += loss.item()

    meanLossPerBatch = totalLoss / batchSize
    return meanLossPerBatch

if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Operate under {device}")

    preprocessor = Preprocessor()
    fps_train, logP_train, fps_validation, logP_validation, fps_test, logP_test = preprocessor.train_test_split()

    trainDataset = CustomDataset(npData=fps_train, npLabel=logP_train, is_test=False)
    validDataset = CustomDataset(npData=fps_validation, npLabel=logP_validation, is_test=False)
    testDataset = CustomDataset(npData=fps_test, npLabel=logP_test, is_test=True)

    input_size = trainDataset.npData.shape[1]

    trainDataloader = DataLoader(dataset=trainDataset, batch_size=100, shuffle=True)
    validDataloader = DataLoader(dataset=validDataset, batch_size=100, shuffle=False)
    testDataloader = DataLoader(dataset=testDataset, batch_size=100, shuffle=False)

    net = MLP(input_size=input_size, out_size = 1).to(device)

    # Train
    epochs = 100
    criterion = nn.MSELoss()
    optimizer = optim.Adam(params = net.parameters(), lr = 0.001)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.95)

    for epoch in range(epochs):
        print(f"Epoch: {epoch+1} \n----------")
        meanLoss = train(net, optimizer, criterion, trainDataloader)
        scheduler.step()
        print(f"meanLoss: {meanLoss}")
        if (epoch % 10 == 0):
            meanLoss = test(net, criterion, validDataloader)
            print(f"valid meanLoss: {meanLoss}")
        
    # Evaluation
    meanLoss = test(net, criterion, testDataloader)
    print(f"Result meanLoss: {meanLoss}")





