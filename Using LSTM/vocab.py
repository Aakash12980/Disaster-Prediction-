
"""

Usage:
    vocab.py --data-path=<file> [options] VOCAB_FILE

Options:
    -h --help                  Show this screen.
    --data-path=<file>         File of training source sentences
    --freq-cutoff=<int>        frequency cutoff [default: 2]
"""


from utils import pad_sent, read_corpus
import torch
from collections import Counter
from itertools import chain
from docopt import docopt
import json
import pickle
from typing import List

class VocabEntry():
    def __init__(self, word2id=None):

        if word2id:
            self.word2id = word2id
        else:
            self.word2id = dict()
            self.word2id["<unk>"] = 0
            self.word2id["<s>"] = 1
            self.word2id["<pad>"] = 2
        self.unk_id = self.word2id["<unk>"]
        self.id2word = {k: v for v,k in self.word2id.items()}

    def __getitem__(self, word):
        # returns the index of the word
        return self.word2id.get(word, self.unk_id) 

    def __contains__(self, word):
        return word in self.word2id

    def __setitem__(self, key, val):
        raise ValueError("Vocabulary is read only")

    def __len__(self):
        return len(self.word2id)

    def __repr__(self):
        return 'Vocabulary[size=%d]' % len(self)

    def id2word(self, idx):
        return self.id2word[idx]
    
    def add(self, word):
        if word not in self:
            wid = self.word2id[word] = len(self)
            self.id2word[word] = wid
            return wid
        else:
            return self[word]
    
    def word2indices(self, words):
        if type(words[0]) == list:
            return [[self[w] for w in s] for s in words]
        else:
            return [self[w] for w in words] 
    
    def indices2word(self, indices):
        return [ self.id2word[idx] for idx in indices]
    
    def to_input_tensor(self, sents:List[List[str]], device:torch.device) -> torch.Tensor:
        word2ids = self.word2indices(sents)
        sents_t = pad_sent(word2ids, self["<pad>"])
        sents_var = torch.tensor(sents_t, dtype=torch.long, device=device)
        return torch.t(sents_var)
    

    @staticmethod
    def from_corpus(corpus, freq_cutoff=2):
        #corpus is train_x returned by read_corpus function of utils.py
        vocab_entry = VocabEntry()
        word_freq = Counter(chain(*corpus))
        valid_words = [w for w,v in word_freq.items() if v >= freq_cutoff]
        print('number of word types: {}, number of word types w/ frequency >= {}: {}'
              .format(len(word_freq), freq_cutoff, len(valid_words)))
        
        sorted_words = sorted(valid_words, key=lambda e: word_freq[e], reverse=True)
        for w in sorted_words:
            vocab_entry.add(w)
        return vocab_entry

class Vocab():
    def __init__(self, x):
        # x is vocabentry object and y is a list of corresponding labels
        self.x = x
    
    @staticmethod
    def build(x, freq_cutoff):
        vocab_entry = VocabEntry.from_corpus(x, freq_cutoff)
        return Vocab(vocab_entry)

    def save(self, file_path):
        json.dump(dict(text=self.x.word2id), open(file_path, 'w'), indent=2)

    @staticmethod
    def load(file_path):
        entry = json.load(open(file_path, 'r'))
        x_word2id = entry['text']
        return Vocab(VocabEntry(x_word2id))

    def __repr__(self):
        return f'Vocab length = {len(self.y)}'

if __name__ == "__main__":
    args = docopt(__doc__)

    print('filepath for data files: %s ' %args['--data-path'])

    x, y = read_corpus(args['--data-path'])
    vocab = Vocab.build(x, int(args['--freq-cutoff']))
    print('generated vocabulary, source %d words' % (len(vocab.x)))

    vocab.save(args['VOCAB_FILE'])
    print('vocabulary saved to %s' % args['VOCAB_FILE'])






             
        

