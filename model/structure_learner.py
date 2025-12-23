import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.util import knn_fast


class StructureLearner(nn.Module):
    def __init__(self, edge_dropout):
        super(StructureLearner, self).__init__()
        self.edge_dropout = edge_dropout

    def graph_process(self, k, x):
        device = x.device
        # 1. 获取原始 KNN 结果 (未归一化)
        rows, cols, values = knn_fast(x, k, 1000)      

        values[torch.isnan(values)] = 0  

        # 2. 对称化 (Symmetrization) [cite: 157]
        rows_ = torch.cat((rows, cols))
        cols_ = torch.cat((cols, rows))
        values_ = torch.cat((values, values))

        # 3. 激活 (Activation) [cite: 157]
        values_ = F.relu(values_)
        
        # Dropout (仅训练时)
        values_ = F.dropout(values_, p=self.edge_dropout, training=self.training)

        num_nodes = x.shape[0]
        indices = torch.stack([rows_, cols_])
        learned_adj = torch.sparse_coo_tensor(
            indices, 
            values_, 
            (num_nodes, num_nodes), 
            device=device
        ).coalesce()

        return learned_adj

    