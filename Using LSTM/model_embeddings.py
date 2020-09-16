import torch.nn as nn

class ModelEmbeddings(nn.Module):
    def __init__(self, embed_size, vocab):
        super(ModelEmbeddings, self).__init__()

        self.embed_size = embed_size

        self.pad_token = vocab.x["<pad>"]
        self.embed_x = nn.Embedding(len(vocab.x), self.embed_size, padding_idx=self.pad_token)
