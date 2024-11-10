from helpers import read_config, singleton

from sentence_transformers import SentenceTransformer
import torch

config = read_config()

@singleton
class SentenceEmbedder:
    def __init__(self, config) -> None:
        # self.device = "torch" if torch.cuda.is_available() else "cpu
        self.device = "cpu"
        self.model = SentenceTransformer(config["rag"]["embedder"], trust_remote_code=True).to(self.device)

    def get_embeddings(self, text):
        return self.model.encode(text, convert_to_tensor=True)
    

if __name__ == "__main__":
    text = "Hello, how are you?"
    embedder = SentenceEmbedder(config)
    embeddings = embedder.get_embeddings(text)
    print(embeddings)
    print(len(embeddings))