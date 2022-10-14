
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
import matplotlib.pyplot as plt
import torch.nn.functional as F
import random
import math
from sklearn.preprocessing import LabelBinarizer


def random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

random_seed(777)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


train_data_path = pd.read_csv("train.csv")
test_data_path = pd.read_csv("test.csv")



def tokenize(path):
    data_frame = path
    data_columns = data_frame.columns.tolist()
    data_columns = [int(i) for i in data_columns]
    data_frame.columns = data_columns
    traindata = data_frame[data_frame.columns[2:]]
    trainlabel = data_frame[data_frame.columns[0]]
    proBert_seq = data_frame[data_frame.columns[1]]
    return np.array(traindata),np.array(trainlabel), np.array(proBert_seq) 

train_X,train_Y, proBer_train_seq  = tokenize(train_data_path)   
testX,testY, proBer_test_seq = tokenize(test_data_path)



class AQYDataset(Dataset):
    def __init__(self, df,train,label, device):

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
        

        # print('label index',label)

        return  sample, seq_feature, label


    def __len__(self):      
      return len(self.protein_seq)



def fit(model, train_loader, optimizer, criterion, device):

    model.train()

    pred_list = []
    label_list = []


    for samples, launch_seq,  label in train_loader: 


        input_ids = samples['input_ids'].to(device)
        token_type_ids = samples['token_type_ids'].to(device)
        attention_mask = samples['attention_mask'].to(device)
        
        
        launch_seq = launch_seq.long().to(device)
        label = torch.tensor(label).float().to(device)
        # label = torch.tensor(label).long().to(device)


        pred = model( input_ids, token_type_ids, attention_mask, launch_seq)

        pred = pred.squeeze()

        loss = criterion(pred, label)
        loss.backward()  
        optimizer.step()
        model.zero_grad()

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
        # label = torch.tensor(label).long().to(device)

        pred = model(input_ids, token_type_ids, attention_mask, launch_seq)
        # print('pred(())))()()()()()',pred)
        
        pred_list.extend(pred.squeeze().cpu().detach().numpy()) 
        label_list.extend(label.squeeze().cpu().detach().numpy()) 

    score = cal_score(pred_list, label_list)
    
    return score

  
#网络模型

tokenizer = BertTokenizer.from_pretrained('Rostlab/prot_bert')
 
class BiLSTM_Attention(nn.Module):

    def __init__(self,  embedding_dim=83,hidden_dim=32,  n_layers=1): 
        super(BiLSTM_Attention, self).__init__()
        self.bert = BertModel.from_pretrained("Rostlab/prot_bert")      
        self.conv1 = nn.Conv1d(41, 16, kernel_size=3, stride=1, padding='same')
#         self.conv2 = nn.Conv1d(16, 8, kernel_size=3, stride=1, padding='same')
        self.n_layers = n_layers
        # self.embedding = nn.Embedding(vocab_size, embedding_dim)        
        self.lstm1 = nn.LSTM(embedding_dim, 64, num_layers=n_layers, bidirectional=True, batch_first=True)
        self.lstm2 = nn.LSTM(64*2, 32, num_layers=n_layers, bidirectional=True, batch_first=True)
        # self.lstm3 = nn.LSTM(2*16, 8, num_layers=n_layers, bidirectional=True, batch_first=True)
        # self.lstm4 = nn.LSTM(2*32, 16, num_layers=n_layers, bidirectional=True, batch_first=True)
        self.fc1 =  nn.Linear(1024+32*2, 16)
        self.fc2 = nn.Linear(16, 5)
        self.fc = nn.Linear(5, 1)     
        self.Rrelu = nn.ReLU()
        self.LRrelu = nn.LeakyReLU()
        self.dropout = nn.Dropout(0.2)
        self.dropout1 = nn.Dropout(0.2)
        self.dropout2 = nn.Dropout(0.2)
        self.w_omega = nn.Parameter(torch.Tensor(hidden_dim * 2+1024, hidden_dim * 2+1024))
        self.u_omega = nn.Parameter(torch.Tensor(hidden_dim * 2+1024, 1))
        nn.init.uniform_(self.w_omega, -0.1, 0.1)
        nn.init.uniform_(self.u_omega, -0.1, 0.1)


    def attention_net(self, x):      
        u = torch.tanh(torch.matmul(x, self.w_omega))         
        att = torch.matmul(u, self.u_omega)                  
        att_score = F.softmax(att, dim=1)
        scored_x = x * att_score                              
        context = torch.sum(scored_x, dim=1)                 
        return context




    def forward(self, input_ids, token_type_ids, attention_mask, x):
       

        pooled_output, _ = self.bert(
            input_ids=input_ids,
            token_type_ids=token_type_ids,
            attention_mask=attention_mask,
            return_dict = False
        )

    
        conv1_output = self.conv1(pooled_output)    
#         conv2_output = self.conv2(conv1_output)
#         print('conv_output shape',conv_output.shape)
        pooled_output = torch.mean(conv1_output, axis=1)  



        output, (final_hidden_state, final_cell_state) = self.lstm1(x.float())  
        output = self.dropout1(output)

        lstmout2, (_, _)  =  self.lstm2(output)   

        bi_lstm_output = self.dropout2(lstmout2)
        bi_lstm_output = torch.mean(bi_lstm_output, axis=1) 
 
        fusion_output = torch.cat([pooled_output, bi_lstm_output], axis=1)   
        fusion_output = fusion_output.unsqueeze(1)

        # lstmout2, (_, _)  =  self.lstm2(output)
        # output = self.dropout(lstmout2)

        # # print('output shape',output.shape) 

        # lstmout3, (_, _)  =  self.lstm3(output)
        # output = self.dropout(lstmout3)
        
        attn_output = self.attention_net(fusion_output)
        out1 = self.fc1(fusion_output)
        out2 = self.fc2(out1)
        logit = self.fc(out2)
        return nn.Sigmoid()(logit)




import copy
import os
import warnings
import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import DataLoader
from sklearn.model_selection import KFold,RepeatedKFold

warnings.filterwarnings('ignore')


X_train = train_X
Y_train = train_Y


skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=123)  


valid_pred=[]
valid_label=[]

independ_pred=[]
independ_label=[]


for index, (train_idx, val_idx) in enumerate(skf.split(X_train,Y_train)):

  
  print('**'*10,'第', index+1, '折','ing....', '**'*10)

  x_train, x_valid = X_train[train_idx], X_train[val_idx]
  y_train, y_valid = Y_train[train_idx],Y_train[val_idx]

  pro_train, pro_valid = proBer_train_seq[train_idx], proBer_train_seq[val_idx]

  

  train_dataset = AQYDataset(pro_train, x_train, y_train, device)
  valid_dataset = AQYDataset(pro_valid, x_valid, y_valid, device)

  test_dataset = AQYDataset(proBer_test_seq, testX, testY, device)


  test_loader = DataLoader(test_dataset,
                batch_size=12,
                shuffle=False,
                drop_last=True,
                num_workers=4)

  
  train_loader = DataLoader(train_dataset,
              batch_size=12,
              shuffle=True,
              drop_last=True,
              num_workers=4)
  
  valid_loader = DataLoader(valid_dataset,
                batch_size=12,
                shuffle=False,
                drop_last=True,
                num_workers=4)
  


  model = BiLSTM_Attention().to(device)
  
  optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=5e-05)  
  
  criterion = nn.BCELoss()



  best_val_score = float('-inf')
  last_improve = 0
  best_model = None

  for epoch in range(20): 
      train_score = fit(model, train_loader, optimizer, criterion, device)
      val_score, _, _ = validate(model, valid_loader, device)
      test_score, _, _  = validate(model, test_loader, device)

      if test_score > best_val_score:
          best_val_score = test_score
          best_model = copy.deepcopy(model)
          last_improve = epoch
          improve = '*'
      else:
          improve = ''

      print(
          f'Epoch: {epoch} Train Score: {train_score}, Valid Score: {val_score} , independ Score:{test_score} {improve} '
      )

  model = best_model

                           
  print(f"=============end!!!!================")
  print("train")
  train_score, _, _ = validate(model, train_loader, device)
  print("valid")
  valid_score, pred_list, label_list = validate(model, valid_loader, device)

  valid_pred.extend(pred_list)
  valid_label.extend(label_list)

  
print("******************************************10-fold cross valid**********************************************")

print("10_cross_valid_score")
cross_valid_score = cal_score(valid_pred,valid_label)




