
from model import ClassifyModel
from utils import read_corpus, DisasterDataset
from model_embeddings import ModelEmbeddings
from vocab import Vocab, VocabEntry
import time
from torch.utils.data import DataLoader
import torch
import torch.nn as nn

TRAIN_BATCH_SIZE = 64
EMBED_DIM = 256
OUTPUT_DIM = 1
N_LAYERS = 1
VOCAB_PATH = './data/vocab.json'
OUTPUT_PATH = './output/models'
TRAIN_PATH = './data/train_xy'
VALID_PATH = './data/valid_xy'
# OUTPUT_DIR = 
DROP_OUT = 0.3
HIDDEN_SIZE = 100
LEARNING_RATE = 1e-4

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def accuracy(preds, y):
    rounded_preds = torch.round(preds)
    correct = (rounded_preds == y).float()
    acc = correct.sum()/len(y)
    return acc

def epoch_time(start_time, end_time):
    time_elapsed = end_time - start_time
    elapsed_min = int(time_elapsed/60)
    elapsed_sec = int(time_elapsed - elapsed_min*60)
    return elapsed_min, elapsed_sec

def collate_fn(batch):
    data_list, label_list = [], []
    for _data, _label in batch:
        data_list.append(_data)
        label_list.append(_label)
    return data_list, label_list

def train(model, iterator, optimizer, criterion):
    epoch_loss = 0
    epoch_acc = 0
    model.train()

    for x_batch, y_batch in iterator:
        data_tuple = zip(x_batch, y_batch)
        sorted_tuple = sorted(data_tuple, key=lambda e: len(e[0]), reverse=True)
        x_batch, y_batch = zip(*sorted_tuple)
        x_batch, y_batch = list(x_batch), list(y_batch)
        y_batch = torch.t(torch.tensor([y_batch], dtype=torch.float32, device = device))

        optimizer.zero_grad()
        x_len = len(x_batch)
        prediction = model(x_batch, x_len)
        # print(f'y_pred: {prediction}\n y_labels: {y_batch}')
        loss = criterion(prediction, y_batch)
        acc = accuracy(prediction, y_batch)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
        epoch_acc += acc.item()
    return epoch_loss/len(iterator), epoch_acc/len(iterator)

def eval(model, iterator, criterion):
    epoch_loss = 0
    epoch_acc = 0
    model.eval()

    for x_iter, y_iter in iterator:
        data_tuple = zip(x_iter, y_iter)
        sorted_tuple = sorted(data_tuple, key=lambda e: len(e[0]), reverse=True)
        x_iter, y_batch = zip(*sorted_tuple)
        x_iter, y_batch = list(x_iter), list(y_iter)
        y_iter = torch.t(torch.tensor([y_iter], dtype=torch.float32, device = device))
        
        x_len = len(x_iter)
        preds = model(x_iter, x_len)
        loss = criterion(preds, y_iter)
        acc = accuracy(preds, y_iter)

        epoch_acc += acc
        epoch_loss += loss
    return epoch_loss/len(iterator), epoch_acc/len(iterator)

def main():
    train_data = read_corpus(TRAIN_PATH)
    eval_data = read_corpus(VALID_PATH)

    # train_data = list(zip(x_train_data, y_train_data))
    # eval_data = list(zip(x_eval_data, y_eval_data))

    vocab = Vocab.load(VOCAB_PATH)
    model = ClassifyModel(EMBED_DIM, HIDDEN_SIZE, OUTPUT_DIM, N_LAYERS, vocab, device, DROP_OUT)

    criterion = nn.BCEWithLogitsLoss()
    model.to(device)
    criterion.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    n_epoch = 3
    train_dataset = DisasterDataset(train_data, device)
    eval_dataset = DisasterDataset(eval_data, device)
    n_iters_train = DataLoader(train_dataset, batch_size=TRAIN_BATCH_SIZE, collate_fn = collate_fn, shuffle=True)
    n_iters_eval = DataLoader(eval_dataset, batch_size=TRAIN_BATCH_SIZE, collate_fn=collate_fn,shuffle=True)

    best_valid_loss = float('inf')

    for epoch in range(n_epoch):
        start_time = time.time()

        train_loss, train_acc = train(model, n_iters_train, optimizer, criterion)
        eval_loss, eval_acc = eval(model, n_iters_eval, criterion)

        end_time = time.time()

        epoch_min, epoch_sec = epoch_time(start_time, end_time)
        if eval_loss < best_valid_loss:
            best_valid_loss = eval_loss
            torch.save(model.state_dict(), 'new_model.pt')
        
        print(f'Epoch: {epoch+1:02} | Epoch Time: {epoch_min}m {epoch_sec}s')
        print(f'\tTrain Loss: {train_loss:.3f} | Train Acc: {train_acc*100:.2f}%')
        print(f'\t Val. Loss: {eval_loss:.3f} |  Val. Acc: {eval_acc*100:.2f}%')



if __name__ == '__main__':
    main()

# model.load_state_dict(torch.load('tut2-model.pt'))

# test_loss, test_acc = evaluate(model, test_iterator, criterion)

# print(f'Test Loss: {test_loss:.3f} | Test Acc: {test_acc*100:.2f}%')
# import spacy
# nlp = spacy.load('en')

# def predict_sentiment(model, sentence):
#     model.eval()
#     tokenized = [tok.text for tok in nlp.tokenizer(sentence)]
#     indexed = [TEXT.vocab.stoi[t] for t in tokenized]
#     length = [len(indexed)]
#     tensor = torch.LongTensor(indexed).to(device)
#     tensor = tensor.unsqueeze(1)
#     length_tensor = torch.LongTensor(length)
#     prediction = torch.sigmoid(model(tensor, length_tensor))
#     return prediction.item()

# predict_sentiment(model, "This film is terrible")