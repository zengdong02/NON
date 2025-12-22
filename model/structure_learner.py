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
        if self.training:
            values_ = F.dropout(values_, p=self.edge_dropout, training=True)

        num_nodes = x.shape[0]
        indices = torch.stack([rows_, cols_])

        # 构建稀疏矩阵并合并重复边 (Coalesce)
        learned_adj = torch.sparse_coo_tensor(
            indices, 
            values_, 
            (num_nodes, num_nodes), 
            device=device
        )
        learned_adj = learned_adj.coalesce()

        indices = learned_adj.indices()
        values = learned_adj.values()
        
        # 利用 sparse sum 或者 scatter_add 计算度向量
        # D[i] = sum_j A_ij
        row_indices = indices[0]
        deg = torch.zeros(num_nodes, device=device)
        deg.scatter_add_(0, row_indices, values)
        
        # 计算 D^{-1/2}
        deg_inv_sqrt = deg.pow(-0.5)
        # 处理除以 0 的情况 (孤立点)
        deg_inv_sqrt.masked_fill_(deg_inv_sqrt == float('inf'), 0)
        
        row, col = indices
        norm_values = values * deg_inv_sqrt[row] * deg_inv_sqrt[col]
        
        # 重新打包成稀疏矩阵
        learned_adj = torch.sparse_coo_tensor(
            indices, 
            norm_values, 
            (num_nodes, num_nodes), 
            device=device
        )

        return learned_adj

    