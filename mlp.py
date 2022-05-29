import torch
import torch.nn as nn
import torch.nn.functional as F
import random
import numpy as np

class MLP(nn.Module):

    def __init__(self, seed, input_dim, dropout_rate):
        super(MLP, self).__init__()
        self.input = nn.Linear(in_features=input_dim,
                                out_features=64)
        self.hidden = nn.Linear(in_features=64,
                                out_features=128)
        self.output = nn.Linear(in_features=128,
                                out_features=10)
        self.dropout = nn.Dropout(p=dropout_rate)

        ## making sure the experiment is reproducible
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True

    def forward(self, x):
        x = F.relu(self.input(x))
        x = self.dropout(x)
        x = F.relu(self.hidden(x))
        x = self.dropout(x)
        x = self.output(x)
        return x