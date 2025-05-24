import os, sys
sys.path.append(os.getcwd())
from omegaconf import OmegaConf
from model.Context_embedder import ContextEmbedder
from datasets import load_dataset
from loss.VAE_loss import LossWithLPIPS

# 加载配置文件
cfg_model = OmegaConf.load('./config/ldm_cond.yaml')
cfg_data = OmegaConf.load('./config/dataset_mscoco.yaml')

# 下载数据集
download_dir = cfg_data.dataset.cache_dir
os.makedirs(download_dir, exist_ok=True)  # 确保目录存在
ds = load_dataset(cfg_data.dataset.name, cache_dir=download_dir)

# 下载预训练模型
embedder = ContextEmbedder(**cfg_model.model.ldm.context_embedder)
loss = LossWithLPIPS()
