import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.util import knn_fast


class StructureLearner(nn.Module):
    def __init__(self, edge_dropout, is_training):
        super(StructureLearner, self).__init__()
        self.edge_dropout = edge_dropout
        self.training = is_training

    def graph_process(self, k, x):
        device = x.device
        rows, cols, values = knn_fast(x, k, 1000)      

        values[torch.isnan(values)] = 0  

        rows_ = torch.cat((rows, cols))
        cols_ = torch.cat((cols, rows))
        values_ = torch.cat((values, values))

        values_ = F.relu(values_)
        values_ = F.dropout(values_, p=self.edge_dropout, training=self.training)

        num_nodes = x.shape[0]

        indices = torch.stack([rows_, cols_])

        learned_adj = torch.sparse_coo_tensor(
            indices, 
            values_, 
            (num_nodes, num_nodes), 
            device=device
        )

        learned_adj = learned_adj.coalesce()

        return learned_adj

    