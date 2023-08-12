import numpy as np
import matplotlib.pyplot as plt
plt.switch_backend('agg')
from utils import read_ZINC
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader


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


class MLP(nn.Module):
    def __init__(self, input_size, out_size, hidden_size=512):
        super(MLP, self).__init__()
        self.hidden_size = hidden_size
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.activation1 = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.activation2 = nn.Tanh()
        self.out = nn.Linear(hidden_size, out_size)

    def forward(self,X):
        out = self.fc1(X)
        out = self.activation1(out)
        out = self.fc2(out)
        out = self.activation2(out)
        out = self.out(out)
        return out
    



if __name__ == "__main__":
    preprocessor = Preprocessor()
    fps_train, logP_train, fps_validation, logP_validation, fps_test, logP_test = preprocessor.train_test_split()


    # Define Model
    model = MLP(fps_train.shape[0], 1)

    # Set an optimizer
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr =0.001)
    
    batch_size = 100
    epoch_size = 100
    batch_train = int(preprocessor.num_train / batch_size)
    batch_validation = int(preprocessor.num_validation / batch_size)
    batch_test = int(preprocessor.num_test / batch_size)

    for i in range(epoch_size):

        ## torch DataLoader에 데이터 넣어야할듯 ==> Preprocessor 클래스에!

