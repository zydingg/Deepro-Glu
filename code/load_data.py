import numpy as np
import pandas as pd




def tokenize(path):
    data_frame = path
    data_columns = data_frame.columns.tolist()
    data_columns = [int(i) for i in data_columns]
    data_frame.columns = data_columns
    traindata = data_frame[data_frame.columns[2:]]
    trainlabel = data_frame[data_frame.columns[0]]
#     proBert_seq = data_frame[data_frame.columns[1]]
    return np.array(traindata),np.array(trainlabel)



train_X,train_Y  = tokenize(train_data_path)   
testX,testY = tokenize(test_data_path)
