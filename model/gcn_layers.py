import torch
import torch.nn as nn


class GCN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(GCN, self).__init__()
        self.fc = nn.Linear(input_dim, output_dim, bias=True)
        self.act = nn.PReLU()
        nn.init.xavier_uniform_(self.fc.weight)
        nn.init.constant_(self.fc.bias, 0.0)

    def forward(self, x, adj):
        x = self.fc(x)
        out = torch.spmm(adj, x)
        return self.act(out)


class GcnLayers(torch.nn.Module):
    def __init__(self, hidden_dim, output_dim, gcn_num_layers, gcn_dropout):
        super(GcnLayers, self).__init__()
        self.num_layers = gcn_num_layers
        self.dropout = nn.Dropout(p=gcn_dropout)
        self.convs = torch.nn.ModuleList()
        self.bns = torch.nn.ModuleList()
        self.sigm = nn.ELU()

        for i in range(self.num_layers):
            in_dim = hidden_dim if i == 0 else output_dim
            self.convs.append(GCN(in_dim, output_dim))
            self.bns.append(nn.BatchNorm1d(output_dim))

    def forward(self, seq, adj):
        graph_output = seq
        for i in range(self.num_layers):
            out = self.convs[i](graph_output, adj)
            if i > 0:
                graph_output = out + graph_output
            else:
                graph_output = out

            graph_output = self.bns[i](graph_output)
            graph_output = self.dropout(graph_output)
            
        return self.sigm(graph_output)

