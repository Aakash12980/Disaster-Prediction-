import numpy as np
import math
import pickle
import torch
from torch.utils.data import Dataset

def pad_sent(sents, pad_token):
    #sents should be list of list of strings(sentences)
    sents_padded = []
    sents_padded = sents.copy()
    sents_lens = [len(s) for s in sents]
    max_len = max(sents_lens)

    pad_list = [[pad_token]*i for i in range(max_len)]
    [sents_padded[idx].extend(pad_list[max_len-i]) for idx, i in enumerate(sents_lens)]

    return sents_padded

def read_corpus(file_path):
    # train_x is list of list of strings and train_y is list of labels
    data = None
    with open(file_path, 'rb') as f:
        train_x, train_y = pickle.load(f)
        data = (train_x, train_y)
    return data

# def batch_iter(data, batch_size, shuffle=False):
#     # data should be list of tuple containing x and y
#     num_batch = math.ceil( len(data) / batch_size )
#     index_array = list(range(len(data)))
#     if shuffle:
#         np.random.shuffle(index_array)
    
#     for i in range(num_batch):
#         indices = index_array[i* batch_size: (i+1)*batch_size]
#         examples = [data[idx] for idx in indices]

#         examples = sorted(examples, key=lambda e: len(e[0]), reverse=True)
#         x_sents = [data[0] for data in examples]
#         y_labels = [data[1] for data in examples]

#         yield x_sents, y_labels

class DisasterDataset(Dataset):
    
    def __init__(self, data, device):
        super(DisasterDataset, self).__init__()
        self.x = data[0]
        self.y = data[1]
        self.n_samples = len(self.x)

    def __getitem__(self, index):
        return self.x[index], self.y[index]

    def __len__(self):
        return self.n_samples



