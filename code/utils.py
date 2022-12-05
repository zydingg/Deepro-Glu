import torch
import torch.nn as nn
import os
import re
import copy
import joblib
import random
import math
import warnings
from torch.utils.data import DataLoader, TensorDataset, Dataset
import warnings
import numpy as np
import pandas as pd
from torch.autograd import Variable
import torch.optim as optim
import matplotlib.pyplot as plt
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset, Dataset
from sklearn.preprocessing import LabelBinarizer, LabelEncoder
from transformers import BertModel, BertTokenizer, AdamW, get_linear_schedule_with_warmup, AutoConfig
from sklearn.metrics import confusion_matrix, auc, roc_curve, roc_auc_score
from sklearn.model_selection import KFold, RepeatedKFold

warnings.filterwarnings('ignore')


def random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


random_seed(777)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def Model_Evaluate(confus_matrix):
    TN, FP, FN, TP = confus_matrix.ravel()

    SN = TP / (TP + FN)
    SP = TN / (TN + FP)
    ACC = (TP + TN) / (TP + TN + FN + FP)
    MCC = ((TP * TN) - (FP * FN)) / (np.sqrt((TP + FN) * (TP + FP) * (TN + FP) * (TN + FN)))
    Pre = TP / (TP + FP)

    return SN, SP, ACC, MCC, Pre


def cal_score(pred, label):
    try:
        AUC = roc_auc_score(list(label), pred)
    except:
        AUC = 0

    pred = np.around(pred)
    label = np.array(label)

    confus_matrix = confusion_matrix(label, pred, labels=None, sample_weight=None)
    SN, SP, ACC, MCC, Pre = Model_Evaluate(confus_matrix)
    print(
        "Model score --- SN:{0:.3f}       SP:{1:.3f}       ACC:{2:.3f}       MCC:{3:.3f}      Pre:{4:.3f}   AUC:{5:.3f}".format(
            SN, SP, ACC, MCC, Pre, AUC))

    return ACC


class AQYDataset(Dataset):
    def __init__(self, df, train, label, device):
        self.protein_seq = df

        self.seq_list = train
        self.label_list = label

    def __getitem__(self, index):
        seq = self.protein_seq[index]
        seq_len = len(seq)
        seq = seq.replace('', ' ')
        encoding = tokenizer.encode_plus(
            seq,
            add_special_tokens=True,
            max_length=41,
            return_token_type_ids=True,
            pad_to_max_length=True,
            return_attention_mask=True,
            return_tensors='pt',
            padding='max_length',
            truncation=True,
        )

        sample = {
            'input_ids': encoding['input_ids'].flatten(),
            'token_type_ids': encoding['token_type_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),

        }

        seq_feature = self.seq_list[index]
        label = self.label_list[index]

        return sample, seq_feature, label

    def __len__(self):
        return len(self.protein_seq)


def fit(model, train_loader, optimizer, criterion, device):
    model.train()

    pred_list = []
    label_list = []

    for samples, launch_seq, label in train_loader:
        input_ids = samples['input_ids'].to(device)
        token_type_ids = samples['token_type_ids'].to(device)
        attention_mask = samples['attention_mask'].to(device)
        launch_seq = launch_seq.long().to(device)
        label = torch.tensor(label).float().to(device)
        pred = model(input_ids, token_type_ids, attention_mask, launch_seq)
        pred = pred.squeeze()
        loss = criterion(pred, label)
        model.zero_grad()
        loss.backward()
        optimizer.step()
        pred_list.extend(pred.squeeze().cpu().detach().numpy())
        label_list.extend(label.squeeze().cpu().detach().numpy())

    score = cal_score(pred_list, label_list)

    return score


def validate(model, val_loader, device):
    model.eval()

    pred_list = []
    label_list = []

    for samples, launch_seq, label in val_loader:
        input_ids = samples['input_ids'].to(device)
        token_type_ids = samples['token_type_ids'].to(device)
        attention_mask = samples['attention_mask'].to(device)
        launch_seq = launch_seq.long().to(device)
        label = torch.tensor(label).float().to(device)
        pred = model(input_ids, token_type_ids, attention_mask, launch_seq)
        pred_list.extend(pred.squeeze().cpu().detach().numpy())
        label_list.extend(label.squeeze().cpu().detach().numpy())

    score = cal_score(pred_list, label_list)

    return score

Deepro_train = joblib.load('./Deepro_Glu_train_pred.pkl')
label = joblib.load('./Deepro_Glu_train_label.pkl')

train_score = cal_score(Deepro_train, label)


