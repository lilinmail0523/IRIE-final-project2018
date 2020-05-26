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
        self.pos, self.node = self.toOneHot(self.data )
    def read_json(self, data_dir):
        with open(data_dir, 'r') as f:
            data_dic = json.loads(f.read())
        return data_dic['Relation']

    def toOneHot(self, data):
        pos = []
        node = []
        for i in data:
            pos.append(i[0][3])
            pos.append(i[1][3])
            node.append(i[0][1])
            node.append(i[1][1])
        pos = pd.Series(pos)
        node = pd.Series(node)

        pos = pd.get_dummies(pos).astype(np.float64)
        node = pd.get_dummies(node).astype(np.float64)

        return pos.values.tolist(), node.values.tolist()

    def __getitem__(self, idx):
        text = self.data[idx]
        pos1 = self.pos[2 * idx]
        pos2 = self.pos[2 * idx + 1]
        node1 = self.node[2* idx]
        node2 = self.node[2* idx+1]

        text1 = text[0][4]
        text2 = text[1][4]
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
    dataset_loader = DataLoader(dataset = nlp_dataset, batch_size = batch_size, shuffle = True)

    return dataset_loader
