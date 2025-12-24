import torch
import torch.nn.functional as F
import random
import os
import numpy as np


def set_seed(seed: int):
    """
    设置全局随机种子，确保实验可复现。
    覆盖: python random, numpy, torch (cpu & gpu), cudnn
    """
    random.seed(seed)
    
    os.environ['PYTHONHASHSEED'] = str(seed)
    
    np.random.seed(seed)
    
    torch.manual_seed(seed)
    
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def to_device(data, device):
    """
    递归地将数据移动到指定设备 (CPU/GPU)。
    支持: torch.Tensor, list, tuple, dict, 以及实现了 .to() 方法的对象 (如 PyG Data)。
    """
    if isinstance(data, torch.Tensor):
        return data.to(device)
    
    elif isinstance(data, dict):
        return {k: to_device(v, device) for k, v in data.items()}
    
    elif isinstance(data, list):
        return [to_device(v, device) for v in data]
    
    elif isinstance(data, tuple):
        return tuple(to_device(v, device) for v in data)
    
    elif hasattr(data, 'to'):
        return data.to(device)
    
    return data


def get_sparse_eye(num_nodes, device='cpu'):
    """
    生成一个形状为 (num_nodes, num_nodes) 的稀疏单位矩阵 (torch.sparse_coo_tensor)
    
    参数:
        num_nodes: 节点数量 (矩阵大小)
        device: 设备 ('cpu' 或 'cuda')
    """
    idx = torch.arange(num_nodes, device=device)
    indices = torch.stack([idx, idx], dim=0)

    values = torch.ones(num_nodes, device=device)

    sparse_eye = torch.sparse_coo_tensor(
        indices, 
        values, 
        (num_nodes, num_nodes), 
        device=device
    ).coalesce()
   
    return sparse_eye


def sim_con(z1, z2, temperature):
    EOS = 1e-10
    z1_norm = torch.norm(z1, dim=-1, keepdim=True)
    z2_norm = torch.norm(z2, dim=-1, keepdim=True)
    dot_numerator = torch.mm(z1, z2.t())
    dot_denominator = torch.mm(z1_norm, z2_norm.t()) + EOS
    sim_matrix = dot_numerator / dot_denominator / temperature
    return sim_matrix


def calc_lower_bound(z_1, z_2, pos, temperature = 0.2):
    EOS = 1e-10    
    if pos.is_sparse:
        pos = pos.to_dense()
    matrix_1 = torch.exp(sim_con(z_1, z_2, temperature))
    matrix_2 = matrix_1.t() 
    
    row_sum_1 = torch.sum(matrix_1, dim=1).view(-1, 1) + EOS
    matrix_1 = matrix_1 / row_sum_1
    probs_1 = matrix_1.mul(pos).sum(dim=-1)
    lori_1 = -torch.log(torch.clamp(probs_1, min=1e-10)).mean()    
    
    row_sum_2 = torch.sum(matrix_2, dim=1).view(-1, 1) + EOS
    matrix_2 = matrix_2 / row_sum_2
    probs_2 = matrix_2.mul(pos).sum(dim=-1)
    lori_2 = -torch.log(torch.clamp(probs_2, min=1e-10)).mean()

    return (lori_1 + lori_2) / 2


def knn_fast(X, k, b):
    device = X.device 
    
    X = F.normalize(X, dim=1, p=2)
    index = 0
    num_nodes = X.shape[0]
    
    values = torch.zeros(num_nodes * (k + 1), device=device)
    rows = torch.zeros(num_nodes * (k + 1), device=device)
    cols = torch.zeros(num_nodes * (k + 1), device=device)

    norm_row = torch.zeros(X.shape[0], device=device)
    norm_col = torch.zeros(X.shape[0], device=device)
    
    while index < num_nodes:
        if (index + b) > num_nodes:
            end = num_nodes
        else:
            end = index + b
            
        sub_tensor = X[index:index + b]
        similarities = torch.mm(sub_tensor, X.t())
        
        # 取 Top K
        vals, inds = similarities.topk(k=k + 1, dim=-1)
        vals = torch.clamp(vals, min=0.0)
        
        start_idx = index * (k + 1)
        end_idx = end * (k + 1)
        
        values[start_idx:end_idx] = vals.view(-1)
        cols[start_idx:end_idx] = inds.view(-1).float()
        
        current_rows = torch.arange(index, end, device=device).view(-1, 1).repeat(1, k + 1).view(-1)
        rows[start_idx:end_idx] = current_rows.float()

        norm_row[index: end] = torch.sum(vals, dim=1)
        norm_col.index_add_(-1, inds.view(-1), vals.view(-1))
        
        index += b
        
    rows = rows.long()
    cols = cols.long()
    norm = norm_row + norm_col
    values *= (torch.pow(norm[rows], -0.5) * torch.pow(norm[cols], -0.5))
    
    return rows, cols, values


def get_k_shot_split(labels, k_shot, num_classes, seed):
    """
    根据给定的 seed 和 k_shot 划分 Support/Query 集合
    
    Args:
        labels (Tensor): 所有节点的标签，Shape [N]
        k_shot (int): 每个类别选多少个样本作为 Support
        num_classes (int): 总类别数
        seed (int): 随机种子
    
    Returns:
        support_idx (Tensor): Support Set 的节点索引
        query_idx (Tensor): Query Set 的节点索引
    """
    # 1. 创建独立的随机状态生成器 (不影响全局随机状态)
    rng = np.random.RandomState(seed)
    
    support_indices = []
    query_indices = []
    
    # 确保 labels 在 CPU 上以便 numpy 处理
    labels_np = labels.cpu().numpy()
    
    for c in range(num_classes):
        # 2. 获取当前类别 c 的所有节点索引
        # np.where 返回的是 tuple，取 [0]
        class_idx = np.where(labels_np == c)[0]
        
        # 3. 检查样本够不够
        if len(class_idx) < k_shot:
            print(f"Warning: Class {c} has only {len(class_idx)} samples, fewer than k={k_shot}. Using all for support.")
            selected = class_idx
            remaining = []
        else:
            # 4. 随机选择 k_shot 个作为 Support
            selected = rng.choice(class_idx, size=k_shot, replace=False)
            # 5. 剩余的作为 Query (使用 setdiff1d 找出差集)
            remaining = np.setdiff1d(class_idx, selected)
            
        support_indices.append(selected)
        query_indices.append(remaining)
    
    # 6. 拼接并转换为 Tensor
    support_idx = np.concatenate(support_indices)
    query_idx = np.concatenate(query_indices)
    
    return torch.from_numpy(support_idx).long(), torch.from_numpy(query_idx).long()


def build_prototypes(embeddings, labels, support_idx, num_classes):
    prototypes = []
    for c in range(num_classes):
        c_mask = (labels[support_idx] == c)
        c_emb = embeddings[support_idx][c_mask]
        
        if c_emb.size(0) == 0:
            # 极少数情况下的防御机制
            proto = torch.zeros(1, embeddings.size(1)).to(embeddings.device)
        else:
            proto = c_emb.mean(dim=0, keepdim=True)
        prototypes.append(proto)
    return torch.cat(prototypes, dim=0) # [Num_Classes, Dim]


def prototypical_loss(prototypes, queries, targets, temperature=0.2):
    prototypes = F.normalize(prototypes, dim=1)
    queries = F.normalize(queries, dim=1)

    logits = torch.mm(queries, prototypes.t()) / temperature
    return F.cross_entropy(logits, targets)


def calculate_acc(prototypes, queries, query_lbl):
    prototypes = F.normalize(prototypes, dim=1)
    queries = F.normalize(queries, dim=1)

    logits = torch.mm(queries, prototypes.t())
    preds = torch.argmax(logits, dim=1)

    acc = (preds == query_lbl).float().mean().item() * 100
    return acc