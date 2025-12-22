import os
import torch
from torch_geometric.datasets import Planetoid, WikipediaNetwork, WebKB
from pathlib import Path
from configs.config import Config

def generate_few_shot_splits(dataset,
                             dataset_name: str,
                             few_shot: int,
                             num_repeat: int,
                             out_dir: str = "splits"):
    
    data = dataset[0] 
    y = data.y.view(-1)

    # 1. 筛选有标签节点
    labeled_mask = y >= 0
    all_labeled_idx = torch.arange(y.size(0))[labeled_mask]
    y_labeled = y[labeled_mask]

    num_classes = int(y_labeled.max().item() + 1)
    
    # 2. 按类别分组索引
    class_to_indices = []
    for c in range(num_classes):
        pos_in_labeled = (y_labeled == c).nonzero(as_tuple=True)[0]
        idx_c = all_labeled_idx[pos_in_labeled]

        if idx_c.numel() < few_shot:
            raise ValueError(
                f"[{dataset_name}] Class {c} has {idx_c.numel()} nodes < {few_shot}-shot."
            )
        class_to_indices.append(idx_c)

    # 3. 生成随机划分
    for r in range(num_repeat):
        torch.manual_seed(Config.seed + r)

        train_idx_parts = [] 
        val_idx_parts = []
        
        num_val = few_shot
        remaining_indices = []

        for idx_c in class_to_indices:
            perm = torch.randperm(idx_c.numel())
            
            chosen_train = idx_c[perm[:few_shot]]
            train_idx_parts.append(chosen_train)
            
            if idx_c.numel() >= few_shot + num_val:
                chosen_val = idx_c[perm[few_shot : few_shot + num_val]]
                val_idx_parts.append(chosen_val)
                chosen_test = idx_c[perm[few_shot + num_val:]]
            else:
                chosen_val = torch.tensor([], dtype=torch.long)
                chosen_test = idx_c[perm[few_shot:]]
                
            remaining_indices.append(chosen_test)

        train_idx = torch.cat(train_idx_parts)
        train_idx, _ = torch.sort(train_idx)
        train_labels = y[train_idx]
        
        if val_idx_parts:
            val_idx = torch.cat(val_idx_parts)
            val_idx, _ = torch.sort(val_idx)
            val_labels = y[val_idx]
        else:
            val_idx = torch.tensor([], dtype=torch.long)
            val_labels = torch.tensor([], dtype=torch.long)

        test_idx = torch.cat(remaining_indices)
        test_idx, _ = torch.sort(test_idx)
        test_labels = y[test_idx]

        split_dir = os.path.join(out_dir, dataset_name, f"{few_shot}_shot", f"split_{r}")
        os.makedirs(split_dir, exist_ok=True)

        torch.save(train_idx, os.path.join(split_dir, f"{dataset_name}_train_idx.pt"))
        torch.save(train_labels, os.path.join(split_dir, f"{dataset_name}_train_labels.pt"))
        
        torch.save(val_idx, os.path.join(split_dir, f"{dataset_name}_val_idx.pt"))
        torch.save(val_labels, os.path.join(split_dir, f"{dataset_name}_val_labels.pt"))
        
        torch.save(test_idx, os.path.join(split_dir, f"{dataset_name}_test_idx.pt"))
        torch.save(test_labels, os.path.join(split_dir, f"{dataset_name}_test_labels.pt"))


if __name__ == "__main__":
    
    THIS_FILE = Path(__file__).resolve()
    PROJECT_ROOT = THIS_FILE.parent.parent
    root = PROJECT_ROOT / "data"

    datasets_config = [
        (Planetoid(root=root, name='Cora'), "Cora"),
        (Planetoid(root=root, name='Pubmed'), "Pubmed"),
        (Planetoid(root=root, name='Citeseer'), "Citeseer"),
        (WikipediaNetwork(root=root, name='Chameleon'), "Chameleon"),
        (WikipediaNetwork(root=root, name='Squirrel'), "Squirrel"),
        (WebKB(root=root, name='Cornell'), "Cornell"),
    ]

    shot_list = [1, 3, 5]
    num_repeat = 50 
    seed = 37
    out_dir_path = PROJECT_ROOT / "data" / "few_shot"

    for shot in shot_list:
        for ds, name in datasets_config:
            try:
                generate_few_shot_splits(
                    dataset=ds,
                    dataset_name=name,
                    few_shot=shot,
                    num_repeat=num_repeat,
                    out_dir=out_dir_path
                )
            except ValueError as e:
                print(f"Skipping {name} for {shot}-shot: {e}")