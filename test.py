from sklearn.metrics import f1_score
import numpy as np
import os
import torch
import time

from model import *
from dataload import *

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')



if __name__ == '__main__':
    data_dir = 'test_processed.json'

    batch_size = 128
    word_embedding_dimension = 300
    POS_Node_dimen = 30
    classes = 3


    #load model
    CNN_classification = classification_net((word_embedding_dimension)*2, classes).to(device)
    model_state_dict = torch.load("SavedModel.tar")
    CNN_classification.load_state_dict(model_state_dict['encoder'])
    print("[Load Model Succeed!]")



    loss_criterion = torch.nn.CrossEntropyLoss()

    start = time.time()
    data_loader = get_loader(data_dir, batch_size)
    loss_total = 0
    y_pred = []
    y_true = []
    for i, (text1, text2, label) in enumerate(data_loader):

        output = CNN_classification(text1, text2)

        loss = loss_criterion(output, label.to(device)).sum()

        loss_total += loss.data.item()
        _, pred = torch.max(output,1)
        y_pred += pred.tolist()
        y_true += label.tolist()
    f1 = f1_score(y_true, y_pred, average = 'micro')
    print('f1 score: ', f1)
    print(loss_total)

