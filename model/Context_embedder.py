from torch import Tensor
from transformers import AutoTokenizer, AutoModel
import torch
from omegaconf import OmegaConf

# 加载配置文件
cfg = OmegaConf.load('./config/ldm_cond.yaml')

class ContextEmbedder:
    def __init__(self, model_name: str, cache_dir: str):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=cache_dir)
        self.model = AutoModel.from_pretrained(model_name, cache_dir=cache_dir)

    def embed(self, text: str) -> Tensor:
        text = 'query:' + text
        input_texts = self.tokenizer(text, max_length=512, padding=True, truncation=True, return_tensors='pt')
        with torch.no_grad():
            model_output = self.model(**input_texts)

        return model_output.pooler_output.squeeze(0)
    
    def embed_batch(self, texts: tuple) -> Tensor:
        """
        为元组中的每个文本添加'query:'前缀并进行批量嵌入
        
        :param texts: 文本元组，大小为batch_size
        :return: 批量文本的嵌入表示
        """
        prefixed_texts = ['query:' + text for text in texts]
        input_texts = self.tokenizer(prefixed_texts, max_length=512, padding=True, truncation=True, return_tensors='pt')
        with torch.no_grad():
            model_output = self.model(**input_texts)
            
        return model_output.pooler_output

    def get_token(self, text: str) -> Tensor:
        text = 'query:' + text
        input_texts = self.tokenizer(text, max_length=512, padding=True, truncation=True, return_tensors='pt')
        return input_texts
    
    def get_batch_token(self, texts: tuple) -> Tensor:
        """
        为元组中的每个文本添加'query:'前缀并获取对应的token
        
        :param texts: 文本元组，大小为batch_size
        :return: 批量文本的token表示
        """
        prefixed_texts = ['query:' + text for text in texts]
        input_texts = self.tokenizer(prefixed_texts, max_length=512, padding=True, truncation=True, return_tensors='pt')
        return input_texts

    def get_token2embedding(self, input_texts: Tensor) -> Tensor:
        with torch.no_grad():
            model_output = self.model(**input_texts)

        return model_output.pooler_output.squeeze(0)
    
    def __call__(self, text):
        """
        使对象可调用，当对象被直接调用时根据输入类型执行相应的嵌入方法
        
        :param text: 输入文本或文本元组
        :return: 文本的嵌入表示
        """
        if isinstance(text, tuple):
            return self.embed_batch(text)
        else:
            return self.embed(text)
        