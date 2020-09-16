import torch
import torch.nn as nn
import sys
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence
from model_embeddings import ModelEmbeddings
from typing import List, Tuple, Dict, Set, Union

class ClassifyModel(nn.Module):
    def __init__(self, embed_size, hidden_size, output_size, n_layers, vocab, device, drop_out=0.2):
        super(ClassifyModel, self).__init__()
        self.model_embed = ModelEmbeddings(embed_size, vocab)
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.drop_out = drop_out
        self.n_layers = n_layers
        self.vocab = vocab
        self.device = device

        self.encoder = nn.LSTM(input_size=embed_size, hidden_size=self.hidden_size, num_layers=self.n_layers, bias=True, bidirectional=False, dropout=self.drop_out)
        self.fully_connected_layer = nn.Linear(in_features=self.hidden_size, out_features=self.output_size, bias=True)
        self.sigmoid_layer = nn.Sigmoid()

    def forward(self, x_data, hidden):
        x_len = [len(i) for i in x_data]

        x_padded = self.vocab.x.to_input_tensor(x_data, device=self.device)

        embed_mat = self.model_embed.embed_x(x_padded)
        seq_padded = pack_padded_sequence(embed_mat, x_len)
        enc_hiddens, (last_hidden, last_cell) = self.encoder(seq_padded)
        enc_hiddens = pad_packed_sequence(sequence=enc_hiddens)[0].permute(1,0,2)
        # concat_layers = torch.cat((last_hidden[-2, :, :], last_hidden[-1, :, :]), 1)
        fully_connect_output = self.fully_connected_layer(last_hidden[0])

        return self.sigmoid_layer(fully_connect_output)


