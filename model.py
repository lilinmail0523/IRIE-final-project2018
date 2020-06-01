import torch
import numpy as np
from gensim.models import KeyedVectors


device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')





class classification_net(torch.nn.Module):
    def __init__(self, embedd_dim ,classes = 3):

        super(classification_net, self).__init__()
        self.conv1 = torch.nn.Conv1d(in_channels = 1, out_channels = 3, kernel_size = 3)

        self.fc1 = torch.nn.Linear(in_features = 600, out_features= 256)
        self.fc2 = torch.nn.Linear(in_features = embedd_dim -2, out_features= classes)
        self.tanh = torch.nn.Tanh()
        self.softmax = torch.nn.Softmax()

    def __init__weight(self):
        self.fc1.weight.data.uniform_(-0.1, 0.1)
        self.fc1.bias.data.fill_(0)
        self.fc2.weight.data.uniform_(-0.1, 0.1)
        self.fc2.bias.data.fill_(0)

    def forward(self, x1, x2):
        #print(x1)
        word1_embedd = torch.stack(x1)
        word2_embedd = torch.stack(x2)

        #print(word1_embedd.shape)

        word1_embedd = word1_embedd.permute(1, 0).to(device)
        word2_embedd = word2_embedd.permute(1, 0).to(device)

        word_concat = torch.cat([word1_embedd,word2_embedd], dim = 1)

        #print(word_concat.shape)
        #x = nn.functional.tanh(self.conv1(word_concat.unsqueeze(1).float()))
        x = word_concat.float().unsqueeze(1)
        x = self.conv1(x)
        x= torch.max(x, 1)[0]
        #print(x.shape)
        x =  self.tanh(x)
        x = self.fc2(x)
        

        x = self.softmax(x)
        return x

