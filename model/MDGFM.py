import torch
import torch.nn as nn
import torch.nn.functional as F
import copy

from .gcn_layers import GcnLayers
from .token_prompt import TokenPrompt
from .structure_learner import StructureLearner
from utils.util import get_sparse_eye, calc_lower_bound


class MDGFM(nn.Module):
    def __init__(self, num_graphs, hidden_dim, output_dim, gcn_num_layers, gcn_dropout, edge_dropout):
        super(MDGFM, self).__init__()
        # Prompt Tokens
        self.pre_tokens = nn.ModuleList([TokenPrompt(hidden_dim) for _ in range(num_graphs)])
        self.global_token = TokenPrompt(hidden_dim)
        self.balance_tokens = nn.ModuleList([TokenPrompt(2 * hidden_dim) for _ in range(num_graphs)])
        # GCN 主干
        self.gcn = GcnLayers(hidden_dim, output_dim, gcn_num_layers, gcn_dropout)

        self.structureLearner = StructureLearner(edge_dropout, True)

    def forward(self, features_list, adj_list, graphs_k):
        x_list = features_list[:]
        num_graphs = len(x_list)

        loss = 0
        for i in range (num_graphs):
            x_list[i] = self.pre_tokens[i](x_list[i])
            x_list[i] = self.global_token(x_list[i])
            hid_x = x_list[i]

            temp_x = torch.sparse.mm(adj_list[i], x_list[i])
            x_list[i] = torch.cat((hid_x, temp_x), dim = 1)
            x_list[i] = self.balance_tokens[i](x_list[i])
            re_adj = self.structureLearner.graph_process(graphs_k[i], x_list[i])

            logits = self.gcn(hid_x, adj_list[i])
            re_logits = self.gcn(hid_x, re_adj)

            pos_eye = get_sparse_eye(logits.shape[0], logits.device)
            loss += calc_lower_bound(logits, re_logits, pos_eye)
            loss += calc_lower_bound(logits, re_logits, re_adj)
        
        return loss

    def get_backbone_model(self, freeze=True):
        gcn_copy = copy.deepcopy(self.gcn)
        
        detached_pre_tokens = [t.token.detach() for t in self.pre_tokens]
        detached_global_token = self.global_token.token.detach()

        if freeze:
            for param in gcn_copy.parameters():
                param.requires_grad = False
            gcn_copy.eval() 

        return detached_pre_tokens, detached_global_token, gcn_copy

        