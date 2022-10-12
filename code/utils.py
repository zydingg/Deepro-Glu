import numpy as np
import torch
from torch.utils.data import Dataset
from tqdm import tqdm
import joblib
from sklearn.metrics import confusion_matrix,auc,roc_curve,roc_auc_score
from sklearn.model_selection import KFold,StratifiedKFold
from sklearn.manifold import TSNE
from transformers import BertModel, BertTokenizer, AdamW, get_linear_schedule_with_warmup, AutoConfig
import seaborn as sns
from torch.autograd import Variable

def Model_Evaluate(confus_matrix):
    TN, FP, FN, TP = confus_matrix.ravel()
    print(TN, FP, FN, TP)
    
    SN = TP / ( TP + FN )
    SP = TN / ( TN + FP )
    ACC = ( TP + TN ) / ( TP + TN + FN + FP )
    MCC = (( TP * TN ) - ( FP * FN )) / (np.sqrt(( TP + FN ) * ( TP + FP ) * ( TN + FP ) * ( TN + FN )))
    Pre = TP/(TP+FP)
    print("Model score --- SN:{0:<20}SP:{1:<20}ACC:{2:<20}MCC:{3:<20}Pre:{4:<20}\n".format(SN, SP, ACC, MCC, Pre))
    
    return SN, SP, ACC, MCC, Pre



def cal_score(pred, label):
    pred = np.array(pred)  
    label = np.array(label)
    
    pred = np.around(pred)
    confus_matrix = confusion_matrix(label, pred, labels=None, sample_weight=None)
    SN, SP, ACC, MCC, Pre = Model_Evaluate(confus_matrix)
    
    return ACC


