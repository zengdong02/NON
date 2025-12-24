import torch
import torch.nn.functional as F
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from torch_geometric.datasets import Planetoid, WikipediaNetwork, WebKB
from torch_geometric.utils import add_self_loops, degree, to_undirected

from config import Config

# 定义全集常量
ALL_DATASETS = Config.graphs_name
GRAPHS_K = Config.graphs_k

def load_processed_data(target_name, root='./data', hidden_dim=50):
    if target_name not in ALL_DATASETS:
        raise ValueError(f"目标必须是以下之一: {ALL_DATASETS}")

    upstream_list = []
    graphs_k = []
    downstream_data = None
    down_k = 15

    for i in range(len(ALL_DATASETS)):
        # 1. 加载原始数据
        name = ALL_DATASETS[i]
        if name in ['Cora', 'Pubmed', 'Citeseer']:
            dataset = Planetoid(root=root, name=name)
        elif name in ['Chameleon', 'Squirrel']:
            dataset = WikipediaNetwork(root=root, name=name)
        elif name == 'Cornell':
            dataset = WebKB(root=root, name=name)
        
        data = dataset[0]

        # SVD 降维 (独立处理，互不干扰)
        # svd = TruncatedSVD(n_components=hidden_dim, random_state=42)
        # reduced_x = svd.fit_transform(data.x.numpy())
        # scaler = StandardScaler()
        # reduced_x = scaler.fit_transform(reduced_x)
        # data.x = torch.from_numpy(reduced_x).float()
        # data.x = F.normalize(data.x, p=2, dim=1)

        # PCA降维
        pca = PCA(n_components=hidden_dim)
        features_np = data.x.numpy()
        reduced_x = pca.fit_transform(features_np)
        data.x = torch.from_numpy(reduced_x).float()

        # 3. 自动分流
        if name == target_name:
            downstream_data = data
            down_k = GRAPHS_K[i]
        else:
            upstream_list.append(data)
            graphs_k.append(GRAPHS_K[i])
    
    return upstream_list, graphs_k, downstream_data, down_k


def get_norm_adj(data):
    num_nodes = data.x.shape[0]
    edge_index = data.edge_index

    edge_index = to_undirected(edge_index, num_nodes=num_nodes)

    edge_index, _ = add_self_loops(edge_index, num_nodes=num_nodes, fill_value=1.0)
    
    row, col = edge_index
    deg = degree(row, num_nodes, dtype=torch.float32)

    deg_inv_sqrt = deg.pow(-0.5)
    deg_inv_sqrt.masked_fill_(deg_inv_sqrt == float('inf'), 0)

    values = deg_inv_sqrt[row] * deg_inv_sqrt[col]

    adj = torch.sparse_coo_tensor(
        indices=edge_index,
        values=values,
        size=(num_nodes, num_nodes)
    ).coalesce()

    return adj


def process_graph_list(data_list):
    """
    输入: 包含 Data 对象的列表 (upstream_list)
    输出: 三个列表 (特征, 标签, 归一化邻接矩阵)
    """
    features_list = []
    labels_list = []
    adj_list = []
    
    for i, data in enumerate(data_list):
        features_list.append(data.x)
        labels_list.append(data.y)
        
        adj = get_norm_adj(data)
        adj_list.append(adj)
                
    return features_list, labels_list, adj_list
