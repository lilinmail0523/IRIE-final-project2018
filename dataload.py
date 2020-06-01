import json
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader



class nlpdataloader(Dataset):
    def __init__(self,  data_dir):
        """

        """
        self.data_dir = data_dir
        self.data = self.read_json(self.data_dir)
    def read_json(self, data_dir):
        with open(data_dir, 'r') as f:
            data_dic = json.loads(f.read())
        return data_dic['Relation']



    def __getitem__(self, idx):
        text = self.data[idx]
        """
        fetch data from preprocessed file
        x[0] : word1 information
        x[1] : word2 information
        x[2] : relation between two word
        word information:
        word[0] : word vector
        word[1] : node vector
        word[2] : location attribute
        word[3] : POS vector
        """

        #0: word vector
        #1: node vector
        #2: location
        #3: POS vector

        text1 = text[0][0] + text[0][1] + text[0][3]
        text2 = text[1][0] + text[1][1] + text[1][3]
        label = text[2]
        if label == 'fact':
            label = 0
        elif label == 'analogy':
            label = 1
        elif label == 'equivalence':
            label = 2



        return text1, text2, label
    def __len__(self):
        return len(self.data)

def get_loader(data_dir, batch_size):
    nlp_dataset = nlpdataloader (data_dir = data_dir)
    print(len(nlp_dataset))
    dataset_loader = DataLoader(dataset = nlp_dataset, batch_size = batch_size, shuffle = True)

    return dataset_loader
