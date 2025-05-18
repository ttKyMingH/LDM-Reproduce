from torch import Tensor
from transformers import AutoTokenizer, AutoModel
import torch

class ContextEmbedder:
    def __init__(self, model_name: str = 'intfloat/multilingual-e5-small'):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)

    def embed(self, text: str) -> Tensor:
        text = 'query:' + text
        input_texts = self.tokenizer(text, max_length=512, padding=True, truncation=True, return_tensors='pt')
        with torch.no_grad():
            model_output = self.model(**input_texts)

        return model_output.pooler_output.squeeze(0)

if __name__ == '__main__':
    embedder = ContextEmbedder('intfloat/multilingual-e5-small')
    cond_embedding = embedder.embed('hello')
    print(cond_embedding.shape)