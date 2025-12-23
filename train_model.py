from utils.load_datasets import load_processed_data, process_graph_list, get_norm_adj
from utils.util import set_seed, to_device, get_k_shot_split, build_prototypes, prototypical_loss
from config import Config
from model.MDGFM import MDGFM
from model.DownModel import DownModel

import torch
import torch.optim as optim
from tqdm import tqdm


def main():

    set_seed(Config.seed)
    device = Config.device

    upstream_list, graphs_k, downstream_data, down_k = load_processed_data(target_name=Config.down_dataset, hidden_dim=Config.hidden_dim)
    
    features_list, labels_list, adj_list = process_graph_list(upstream_list)

    down_features, down_labels, down_adj = downstream_data.x, downstream_data.y, get_norm_adj(downstream_data)

    features_list = to_device(features_list, device)
    labels_list   = to_device(labels_list, device)
    adj_list      = to_device(adj_list, device)

    down_features = to_device(down_features, device)
    down_labels   = to_device(down_labels, device)
    down_adj      = to_device(down_adj, device)

    model = MDGFM(len(upstream_list), Config.hidden_dim, Config.output_dim, Config.gcn_num_layers, Config.gcn_dropout, Config.edge_dropout)
    model.to(device)

    optimiser = optim.Adam(model.parameters(), lr=Config.pre_lr, weight_decay=0.0001)
    # scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimiser, mode='min', factor=0.5, patience=5)

    best_loss = float('inf')
    model_wait = 0
    for epoch in tqdm(range(Config.epochs), desc="Training Progress", ncols=100):
        loss = 0
        model.train()
        optimiser.zero_grad()

        loss = model(features_list, adj_list, graphs_k)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=3.0)
        optimiser.step()

        current_loss = loss.item()
        # scheduler.step(current_loss)
        if current_loss < best_loss:
            best_loss = current_loss
            model_wait = 0
            torch.save(model.state_dict(), Config.save_path)
            tqdm.write(f"Epoch {epoch}: New best loss {best_loss:.4f}, model saved.")
        else:
            model_wait += 1

        if model_wait >= Config.patience:
            tqdm.write(f'Early stopping at epoch {epoch}!')
            break

    del features_list, labels_list, adj_list, model, optimiser
    torch.cuda.empty_cache()

    print('#'*100)
    print('Downastream dataset is ',Config.down_dataset)

    loaded_model = MDGFM(len(upstream_list), Config.hidden_dim, Config.output_dim, Config.gcn_num_layers, Config.gcn_dropout, Config.edge_dropout)
    loaded_model.load_state_dict(torch.load(Config.save_path, map_location=device))

    pre_tokens, global_token, gcn = loaded_model.get_backbone_model(freeze=True)
    num_classes = len(torch.unique(down_labels))
    accs = []
    
    for i in tqdm(range(Config.down_tasks), desc="Down Task Training Progress", ncols=100):
        set_seed(i)
        support_idx, query_idx = get_k_shot_split(
            labels=down_labels, 
            k_shot=Config.k_shot, 
            num_classes=num_classes, 
            seed=i
        )
        support_idx = to_device(support_idx, device)
        query_idx = to_device(query_idx, device)

        support_labels = down_labels[support_idx]

        down_model = DownModel(pre_tokens, global_token, gcn, Config.hidden_dim, num_classes, Config.edge_dropout)
        down_model.to(device)
        down_optimiser = torch.optim.Adam(down_model.parameters(), lr=Config.down_lr, weight_decay=5e-4) 

        for _ in range(Config.down_epoch):
            down_model.train()
            down_optimiser.zero_grad()

            embeddings = down_model(down_features, down_adj, down_k)
            prototypes = build_prototypes(embeddings, down_labels, support_idx, num_classes)

            if Config.k_shot == 1:
                emb_view2 = down_model(down_features, down_adj, down_k)
                queries = emb_view2[support_idx]
            else:
                queries = embeddings[support_idx]

            loss = prototypical_loss(prototypes, queries, support_labels)
            loss.backward()
            down_optimiser.step()
        
        down_model.eval()
        with torch.no_grad():
            embeddings = down_model(down_features, down_adj, down_k)
            final_prototypes = build_prototypes(embeddings, down_labels, support_idx, num_classes)
            query_emb = embeddings[query_idx]
            query_lbl = down_labels[query_idx]  
            
            dists = torch.cdist(query_emb, final_prototypes, p=2)
            preds = torch.argmin(dists, dim=1)
            acc = (preds == query_lbl).float().mean().item() * 100

            accs.append(acc)
            tqdm.write(f"Run {i}, Test Acc: {acc:.4f}")

    print('-' * 100)
    accs_tensor = torch.tensor(accs)
    print('Mean:[{:.4f}]'.format(accs_tensor.mean().item()))
    print('Std :[{:.4f}]'.format(accs_tensor.std().item()))


if __name__ == '__main__':
    main()