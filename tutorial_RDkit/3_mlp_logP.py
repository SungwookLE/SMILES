import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
plt.switch_backend('agg')
from utils import read_ZINC
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem


class Preprocessor:
    def __init__(self):
        self.fps_total, self.logP_total, self.tpsa_total = read_ZINC(60000)
        self.train_test_split()
        return self.fps_train, self.logP_train, self.fps_validation, self.logP_validation, self.fps_test, self.logP_test

      
    def train_test_split(self):
        num_train = 4000
        num_validation = 10000
        num_test = 10000

        self.fps_train = self.fps_total[0:num_train]
        self.logP_train = self.logP_total[0:num_train]
        self.fps_validation = self.fps_total[num_train:(num_train+num_validation)]
        self.logP_validation = self.logP_total[num_train:(num_train+num_validation)]
        self.fps_test = self.fps_total[(num_train+num_validation):]
        self.logP_test = self.logP_total[(num_train+num_validation):]


if __name__ == "__main__":
    fps_train, logP_train, fps_validation, logP_validation, fps_test, logP_test = Preprocessor()
    print(logP_test)