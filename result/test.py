import torch
import torch.nn as nn
import os
from torch.utils.data import DataLoader, TensorDataset, Dataset
import warnings
import re  
import numpy as np
import pandas as pd
import joblib
from torch.autograd import Variable
import torch.optim as optim
import torch.nn.functional as F
import random
import math





new_model = BiLSTM_Attention().to(device)      
new_model.load_state_dict(torch.load("./Deepro_Glu.pkl"))      

test_dataset = AQYDataset(proBer_test_seq, testX, testY, device)
test_loader = DataLoader(test_dataset,
              batch_size=12,
              shuffle=False,
              drop_last=True,
              num_workers=4)


print("test")
test_score, _, _  = validate(new_model, test_loader, device)
