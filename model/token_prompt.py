import torch
import torch.nn as nn
import torch.nn.functional as F

class TokenPrompt(nn.Module):
    def __init__(self, hidden_dim):
        super(TokenPrompt, self).__init__()
        self.token= nn.Parameter(torch.FloatTensor(1,hidden_dim))
        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.xavier_uniform_(self.token)
        # torch.nn.init.ones_(self.token)
        # torch.nn.init.normal_(self.token, mean=1.0, std=0.01)

    def forward(self, x):
        x = self.token * x
        # x = F.normalize(x, p=2, dim=1)
        return x
