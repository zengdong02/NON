# 方法1
import torch
import torch.nn as nn
import torch.nn.functional as F
from models import DGI, GraphCL, Lp
from layers import GCN, AvgReadout
import tqdm
import numpy as np


class GcnLayers(torch.nn.Module):
    def __init__(self, n_in, n_h,num_layers_num,dropout):
        super(GcnLayers, self).__init__()

        self.act=torch.nn.ReLU()
        self.num_layers_num=num_layers_num
        self.g_net, self.bns = self.create_net(n_in,n_h,self.num_layers_num)

        self.dropout=torch.nn.Dropout(p=dropout)

    def create_net(self,input_dim, hidden_dim,num_layers):
        
        self.convs = torch.nn.ModuleList()
        self.bns = torch.nn.ModuleList()
        for i in range(num_layers):

            if i:
                nn = GCN(hidden_dim, hidden_dim)
            else:
                nn = GCN(input_dim, hidden_dim)
            conv = nn
            bn = torch.nn.BatchNorm1d(hidden_dim)

            self.convs.append(conv)
            self.bns.append(bn)

        return self.convs, self.bns


    def forward(self, seq, adj,sparse,LP=False):
        graph_output = torch.squeeze(seq,dim=0)
        graph_len = adj
        xs = []
        for i in range(self.num_layers_num):

            input=(graph_output,adj)
            if i:
                graph_output = self.convs[i](input) + graph_output
            else:
                graph_output = self.convs[i](input)
            if LP:
                graph_output = self.bns[i](graph_output)
                graph_output = self.dropout(graph_output)
            xs.append(graph_output)
            
        return graph_output.unsqueeze(dim=0)
    

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class GCN(nn.Module):
    def __init__(self, in_ft, out_ft, act=None, bias=True):
        super(GCN, self).__init__()
        self.fc = nn.Linear(in_ft, out_ft, bias=False)
        self.act = nn.PReLU()

        
        if bias:
            self.bias = nn.Parameter(torch.FloatTensor(out_ft))
            self.bias.data.fill_(0.0)
        else:
            self.register_parameter('bias', None)

        for m in self.modules():
            self.weights_init(m)

    def weights_init(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight.data)
            if m.bias is not None:
                m.bias.data.fill_(0.0)


    def forward(self, input, sparse=True):

        seq = input[0]
        if isinstance(input[1], np.ndarray):  
            input = (input[0], torch.from_numpy(input[1]).to(input[0].device))            
            adj = input[1]
        else:
            adj = input[1]
        seq_fts = self.fc(seq)
        if sparse:
            adj = adj.float()
            out = torch.spmm(adj, seq_fts)
        else:
            out = torch.mm(adj.squeeze(dim=0), seq_fts)
        if self.bias is not None:
            out += self.bias

        return self.act(out)



# 方法2
import torch
import torch.nn as nn
from torch_geometric.nn import GCNConv


class GcnLayers(torch.nn.Module):
    def __init__(self, input_dim, output_dim, num_layers, dropout):
        super(GcnLayers, self).__init__()

        self.act = torch.nn.PReLU()
        self.dropout = torch.nn.Dropout(p=dropout)
        self.num_layers = num_layers

        self.convs = torch.nn.ModuleList()
        self.bns = torch.nn.ModuleList()

        self._create_net(input_dim, output_dim, num_layers) 
        
    def _create_net(self, input_dim, output_dim, num_layers):
        for i in range(num_layers):
            in_d = input_dim if i == 0 else output_dim
            
            conv = GCNConv(in_d, output_dim, normalize=False, bias=True)
            self.convs.append(conv)

            bn = torch.nn.BatchNorm1d(output_dim)
            self.bns.append(bn)
                

    def forward(self, x, adj):
        edge_index = adj.indices()
        edge_weight = adj.values()
        
        h = x
        for i in range(self.num_layers):
            h_in = h
            h = self.convs[i](h, edge_index, edge_weight)
            h = self.act(h)

            if i > 0:
                h = h + h_in
            h = self.bns[i](h)
            h = self.dropout(h)

        return h