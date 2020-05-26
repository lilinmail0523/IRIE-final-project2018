from sklearn.metrics import f1_score
import numpy as np
import os

import torch
import time

from model import *
from dataload import *


device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')



if __name__ == '__main__':
    batch_size = 128
    epoch = 100
    word_embedding_dimension = 300

    POS_Node_dimen = 30
    classes = 3
    f1MAX = 0

    """
    split data into train/validation
    reference: https://stackoverflow.com/questions/50544730/how-do-i-split-a-custom-dataset-into-training-and-test-datasets
    16959 = len
    """
    data_loader = nlpdataloader('train_processed.json')

    train_size = int(0.8 * len(data_loader))
    val_size = len(data_loader) - train_size

    train_dataset, val_dataset = torch.utils.data.random_split(data_loader, [train_size, val_size])
    
    Train_Data = DataLoader(dataset = train_dataset, batch_size = batch_size, shuffle = True)
    Val_Data=  DataLoader(dataset = val_dataset, batch_size = batch_size, shuffle = True)



    CNN_classification = classification_net((word_embedding_dimension)*2, classes).to(device)
    #dimension_reduc = dimmen_reduction(word_embedding_dimension, word_output_dimension).to(device)


    loss_criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(params=CNN_classification.parameters(), lr=0.005, weight_decay = 1e-5)




    for epoch_id in range(epoch):
        print("==================", epoch_id," epochs==================")
        start = time.time()
        loss_total = 0
        y_pred = []
        y_true = []

        """
        Trainning Process
        """
        CNN_classification.train()

        for i, (text1, text2, label) in enumerate(Train_Data):

            output = CNN_classification(text1, text2)

            loss = loss_criterion(output, label.to(device)).sum()
            optimizer.zero_grad()
            loss.backward()

            optimizer.step()
            loss_total += loss.data.item()
            _, pred = torch.max(output,1)
            y_pred += pred.tolist()
            y_true += label.tolist()
        f1 = f1_score(y_true, y_pred, average = 'micro')
        print('Training f1 score/toatl loss: ', f1, '/',loss_total)

        """
        Validation Process
        """
        CNN_classification.eval()

        with torch.no_grad():
            for i, (text1, text2, label) in enumerate(Val_Data):
                output = CNN_classification(text1, text2)
                loss = loss_criterion(output, label.to(device)).sum()
                loss_total += loss.data.item()
                _, pred = torch.max(output,1)
                y_pred += pred.tolist()
                y_true += label.tolist()

            f1 = f1_score(y_true, y_pred, average = 'micro')
            print('-' * 50)
            print('Validation f1 score:/toatl loss ', f1, '/',loss_total)
            if (f1MAX < f1):
                f1MAX = f1
                torch.save({'encoder': CNN_classification.state_dict(),
						    'optimizer': optimizer.state_dict(),
						    'epoch':epoch_id},
						    "SavedModel.tar")
                print("Save best validation f1 model!")
