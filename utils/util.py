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
    )
   
    return sparse_eye.coalesce()


def inspect_tensor(name, tensor):
    """详细打印 Tensor 的元数据信息"""
    if not torch.is_tensor(tensor):
        print(f"⚠️ [{name}] 不是 Tensor，类型是: {type(tensor)}")
        if isinstance(tensor, (int, float)):
             print(f"   Value: {tensor}")
        return

    # 基础信息
    layout_type = "SPARSE" if tensor.is_sparse else "DENSE"
    info = (
        f"🔍 [{name}] "
        f"Shape={tuple(tensor.shape)} | "
        f"Type={layout_type} ({tensor.layout}) | "
        f"Device={tensor.device} | "
        f"Dtype={tensor.dtype}"
    )

    # 数值健康检查 (NaN/Inf)
    # 注意：稀疏矩阵直接用 .any() 可能会报错或很慢，通常只检查 values
    try:
        if tensor.is_sparse:
            values = tensor.values()
            nnz = tensor._nnz()
            info += f" | NNZ={nnz}" # 非零元素数量
        else:
            values = tensor
        
        has_nan = torch.isnan(values).any().item()
        has_inf = torch.isinf(values).any().item()
        
        if has_nan: info += " | ❌ 含 NaN"
        if has_inf: info += " | ❌ 含 Inf"
        
        # 打印部分统计值帮助判断量级
        if values.numel() > 0 and not has_nan:
             info += f" | Min={values.min().item():.4f}, Max={values.max().item():.4f}"

    except Exception as e:
        info += f" | (数值检查失败: {e})"

    print(info)


def sim_con(z_1, z_2, temperature):
    """
    计算两个特征矩阵之间的余弦相似度 (全稠密计算版本)。
    
    Args:
        z_1: (N, D) Tensor, 节点特征 1
        z_2: (N, D) Tensor, 节点特征 2
        temperature: float, 温度系数 (例如 0.2)
        
    Returns:
        logits: (N, N) Dense Tensor, 相似度矩阵 (未经过 exp)
    """
    # 1. 安全检查：如果输入是稀疏矩阵，强制转为稠密
    #    这样能保证后续的矩阵乘法使用针对稠密优化的 torch.mm
    if z_1.is_sparse:
        z_1 = z_1.to_dense()
    if z_2.is_sparse:
        z_2 = z_2.to_dense()

    # 2. L2 归一化 (L2 Normalization)
    #    余弦相似度 = (A . B) / (|A| * |B|)
    #    先对向量做归一化，之后只需要做点积即可
    z_1_norm = F.normalize(z_1, dim=1)
    z_2_norm = F.normalize(z_2, dim=1)
    
    # 3. 矩阵乘法 (Matrix Multiplication)
    #    (N, D) @ (D, N) -> (N, N)
    #    结果范围通常在 [-1/temp, 1/temp] 之间
    similarity = torch.mm(z_1_norm, z_2_norm.t())
    
    return similarity / temperature

def calc_lower_bound(z_1, z_2, pos, temperature=0.2):
    """
    方法 1: 全稠密计算 (适用于 N < 10000 的场景)
    不管输入是 Sparse 还是 Dense，内部统一转为 Dense 运算，彻底杜绝 Sparse 算子报错。
    """
    EOS = 1e-10
    
    # 1. 统一转为 Dense，确保设备一致
    #    即使 pos 是 sparse，to_dense() 后也就 183x183，非常小
    z_1 = z_1.to_dense() if z_1.is_sparse else z_1
    z_2 = z_2.to_dense() if z_2.is_sparse else z_2
    pos = pos.to_dense() if pos.is_sparse else pos
    
    # 确保在同一设备
    if pos.device != z_1.device:
        pos = pos.to(z_1.device)

    # 2. 计算相似度 (结果必为 Dense)
    #    sim_con 内部可以是简单的 (z1 @ z2.T) / temp
    sim_matrix = torch.exp(sim_con(z_1, z_2, temperature))

    # 3. Lori 1 (行归一化)
    #    Dense / Dense -> Broadcasting 完美支持
    row_sum = sim_matrix.sum(dim=1, keepdim=True) + EOS
    prob_1 = sim_matrix / row_sum
    
    #    element-wise 乘法 -> 求和 -> log
    lori_1 = -torch.log(torch.clamp(prob_1.mul(pos).sum(dim=-1), min=EOS)).mean()

    # 4. Lori 2 (列归一化)
    col_sum = sim_matrix.sum(dim=0, keepdim=True) + EOS
    prob_2 = sim_matrix / col_sum
    
    #    注意：这里 prob_2 需要转置来匹配 pos 的行
    #    或者：pos.t() * prob_2 (取决于你的数学定义，通常是对称的)
    #    根据你之前的代码逻辑 prob_2 = prob_2.t()
    prob_2 = prob_2.t()
    
    lori_2 = -torch.log(torch.clamp(prob_2.mul(pos).sum(dim=-1), min=EOS)).mean()

    return (lori_1 + lori_2) / 2


def knn_fast(X, k, b):
    device = X.device 
    
    X = F.normalize(X, dim=1, p=2)
    index = 0
    num_nodes = X.shape[0]
    
    values = torch.zeros(num_nodes * (k + 1), device=device)
    rows = torch.zeros(num_nodes * (k + 1), device=device)
    cols = torch.zeros(num_nodes * (k + 1), device=device)
    norm_row = torch.zeros(num_nodes, device=device)
    norm_col = torch.zeros(num_nodes, device=device)
    
    while index < num_nodes:
        if (index + b) > num_nodes:
            end = num_nodes
        else:
            end = index + b
            
        sub_tensor = X[index:index + b]
        similarities = torch.mm(sub_tensor, X.t())
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
        
    norm = norm_row + norm_col
    rows = rows.long()
    cols = cols.long()

    EOS = 1e-10
    norm = torch.clamp(norm, min=EOS)
    
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
        # 找出当前类别 c 在 support set 中的位置
        c_mask = (labels[support_idx] == c)
        c_emb = embeddings[support_idx][c_mask]
        
        if c_emb.size(0) == 0:
            # 极少数情况下的防御机制
            proto = torch.zeros(1, embeddings.size(1)).to(embeddings.device)
        else:
            proto = c_emb.mean(dim=0, keepdim=True)
        prototypes.append(proto)
    return torch.cat(prototypes, dim=0) # [Num_Classes, Dim]

# --- 辅助函数：原型 Loss ---
def prototypical_loss(prototypes, queries, targets):
    # dists: [Batch, Num_Classes]
    dists = torch.cdist(queries, prototypes, p=2) 
    return F.cross_entropy(-dists, targets)