import os
import time
import torch
import yaml

class Config:
    def __init__(self, config_file="configs/config.yaml"):
        # 1. 获取 config.yaml 的绝对路径
        current_dir = os.path.dirname(os.path.abspath(__file__))
        file_path = os.path.join(current_dir, config_file)

        # 2. 读取 YAML 文件
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"配置文件未找到: {file_path}")

        with open(file_path, 'r', encoding='utf-8') as f:
            config_params = yaml.safe_load(f)

        # 3. 加载通用/环境设置
        self.device_str = config_params.get('device', 'cpu')
        self.device = torch.device(self.device_str)
        self.seed = config_params.get('seed', 1024)
        self.graphs_name = config_params.get('graphs_name')
        self.graphs_k = config_params.get('graphs_k')

        # 自动生成时间戳和保存路径
        self.timestamp = time.strftime("%Y%m%d_%H%M%S")
        self.save_dir = os.path.join(current_dir, "result")
        self.save_path = os.path.join(self.save_dir, f"{self.timestamp}_AGFM.pth")

        # 4. 加载 PreModel 参数
        pre_params = config_params.get('PreModel', {})
        self.pre_lr = pre_params.get('lr', 0.001)
        self.hidden_dim = pre_params.get('hidden_dim', 64) 
        self.output_dim = pre_params.get('output_dim', 256)
        self.gcn_num_layers = pre_params.get('gcn_num_layers', 3)
        self.edge_dropout = pre_params.get('edge_dropout', 0.5)
        self.gcn_dropout = pre_params.get('gcn_dropout', 0.5)
        self.epochs = pre_params.get('epochs', 100)
        self.patience = pre_params.get('patience', 10)

        # 5. 加载 DownModel 参数
        down_params = config_params.get('DownModel', {})
        self.down_dataset = down_params.get('down_dataset', 'Cora')
        self.k_shot = down_params.get('k_shot', 1)
        self.down_tasks = down_params.get('down_tasks', 50)
        self.down_epoch = down_params.get('down_epoch', 100)
        self.down_lr = down_params.get('down_lr', 0.005)

    def __repr__(self):
        return str(self.__dict__)


try:
    Config = Config()
    print("配置加载成功")
except Exception as e:
    print(f"配置加载失败: {e}")
    raise e