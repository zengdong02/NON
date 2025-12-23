import torch
import torch.nn as nn
import torch.nn.functional as F

from .token_prompt import TokenPrompt
from .structure_learner import StructureLearner


class PrePrompt(nn.Module):
    def __init__(self, token_list, global_token, hidden_dim):
        super(PrePrompt, self).__init__()
        self.register_buffer('tokens', torch.cat(token_list, dim=0))
        self.weighted_prompt = WeightedPrompt(len(token_list))
        self.register_buffer('global_token', global_token)
        self.pre_token = TokenPrompt(hidden_dim)
        self.combine_prompt = CombinePrompt()

    def forward(self, seq):
        x = seq
        pre_token = self.weighted_prompt(self.tokens)
        x = pre_token * x
        x = F.relu(x)
        x = self.global_token * x

        x1 = seq
        x1 = self.pre_token(x1)

        x2 = self.combine_prompt(x, x1)
        return x2
    
class CombinePrompt(nn.Module):
    def __init__(self):
        super(CombinePrompt, self).__init__()
        self.weight = nn.Parameter(torch.FloatTensor(1, 2), requires_grad=True)
        self.act = nn.ELU()
        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.constant_(self.weight, 0.5)

    def forward(self, x, x1):
        hid = self.weight[0][0] * x + self.weight[0][1] * x1
        return self.act(hid)

class WeightedPrompt(nn.Module):
    def __init__(self, weighted_num):
        super(WeightedPrompt, self).__init__()
        self.weight= nn.Parameter(torch.FloatTensor(1, weighted_num), requires_grad=True)
        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.xavier_uniform_(self.weight)
        # torch.nn.init.normal_(self.weight, mean=1.0, std=0.01)

    def forward(self, x):
        return torch.mm(self.weight, x)


class DownModel(nn.Module):
    def __init__(self, pre_tokens, global_token, gcn, hidden_dim, num_classes, edge_dropout):
        super(DownModel, self).__init__()

        self.pre_prompt = PrePrompt(pre_tokens, global_token, hidden_dim)
        self.balance_prompt = TokenPrompt(2 * hidden_dim)
        self.gcn = gcn
        self.structureLearner = StructureLearner(edge_dropout)

        self.nb_classes = num_classes
        # self.alpha = nn.Parameter(torch.tensor(0.5))
        self.register_buffer('alpha', torch.tensor(0.5))


    def forward(self, features, adj, down_k):

        features1 = self.pre_prompt(features)

        reseq1 = torch.sparse.mm(adj,features1)
        reseq1 = torch.cat((features1, reseq1), dim = 1)

        reseq111 = self.balance_prompt(reseq1) 

        re_adj = self.structureLearner.graph_process(down_k, reseq111)

        new_adj = (self.alpha * adj + (1 - self.alpha) * re_adj).coalesce()

        embedding = self.gcn(features1, new_adj)      

        return embedding

   