import re
import os
import pickle
import torch
import nltk
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torch.autograd import Variable

class TextProcessor():
    """
    """
    def __init__(self):
        sub_list = [
            (r'<br />', r' '),
            (r'\d+\.?\d+', r'1'),
            (r'[^\w!  ?,.\']', r' '),
            (r'(\.{2,})', r' ... '),
            (r'\.\s', r' . '),
            (r"'s", r" 's"),
            (r'!', r' ! '),
            (r',', r' , '),
            (r'\?', r' ? '),
            (r'\s+', r' '),
        ]
        self.patterns = [(re.compile(x), y) for x, y in sub_list]

        if nltk.download('stopwords'):
            self.stopwords = nltk.corpus.stopwords.words('english')
        else:
            print('***** Warning: Failed to download stopwords! *****')
            self.stopwords = None

        if (nltk.download("wordnet") and nltk.download("punkt") and
            nltk.download("maxent_treebank_pos_tagger") and
            nltk.download("averaged_perceptron_tagger")):
            self.lemmatizer = nltk.stem.WordNetLemmatizer()
        else:
            print('***** Warning: Failed to download lemmatizer! *****')
            self.lemmatizer = None

    def __call__(self, text):
        for pattern, target in self.patterns:
            text = pattern.sub(target, text)
        text = text.lower().strip().split()
        text = self.del_stopwords(text)
        if self.lemmatizer:
            text = self.lemmatizer
        return text

    def del_stopwords(self, text):
        if self.stopwords:
            result = []
            for word in text:
                if word not in self.stopwords:
                    result.append(word)
            return result
        else:
            return text

    def lemmatization(self, text):
        return [self.lemmatizer.lemmatize(x) for x in text]


class Encoder():
    """
    """
    def __init__(self):
        self.encoder = {}
        self.word_count = {}

    def add(self, text):
        if isinstance(text, str):
            text = text.split()
        for word in text:
            if word in self.encoder:
                self.word_count[word] += 1
            else:
                self.encoder[word] = len(self.encoder)+1
                self.word_count[word] = 1

    def __call__(self, word):
        if word in self.encoder:
            return self.encoder[word]
        else:
            return 1

    def clip(self, max_num):
        inorder_list = sorted(self.word_count.items(),
                              key=lambda t:t[1], reverse=True)
        count = 2
        for word, num in inorder_list:
            if count < max_num:
                self.encoder[word] = count
            else:
                self.encoder.pop(word)
                self.word_count.pop(word)
            count += 1


class IMDB(Dataset):
    """
    """
    def __init__(self, train_file, test_file, max_num,
                 seq_length, training=True):
        super(IMDB, self).__init__()
        save_file = 'data.pkl'
        if os.path.exists(save_file):
            with open(save_file, 'rb') as fp:
                self.data, encoder = pickle.load(fp)
        else:
            # loading train data
            with open(train_file, 'r') as fp:
                fp.readline()
                self.data = fp.readlines()

            processor = TextProcessor()
            self.data = [x.split('\t')[1:] for x in self.data]
            self.data = [(processor(y), int(x)) for x, y in self.data]

            encoder = Encoder()
            for text, label in self.data:
                encoder.add(text)

            # loading test data
            with open(test_file, 'r') as fp:
                fp.readline()
                test_data = fp.readlines()

            test_data = [x.split('\t') for x in test_data]
            test_data = [(processor(y), x) for x, y in test_data]
            self.data = (self.data, test_data)

            with open(save_file, 'wb') as fp:
                pickle.dump((self.data, encoder), fp)
        self.data = self.data[0] if training else self.data[1]
        encoder.clip(max_num)
        self.data = [(x[:seq_length], y) for x, y in self.data]
        self.data = [(torch.LongTensor([encoder(w) for w in x] +
                     [0]*(seq_length-len(x))), y) for x, y in self.data]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


def load(train_file, test_file, max_num, seq_length, batch_size=128):
    print('==== Loading IMDB dataset.. ====')
    trainset = IMDB(train_file, test_file, max_num, seq_length, training=True)
    train_loader = DataLoader(trainset, batch_size=batch_size,
                        shuffle=True, num_workers=4)
    testset = IMDB(train_file, test_file, max_num, seq_length, training=False)
    test_loader = DataLoader(testset, batch_size=batch_size,
                        shuffle=False, num_workers=4)
    return train_loader, test_loader


class Model(nn.Module):
    """
    """
    def __init__(self, vocab_size, embeding_dim, hidden_dim):
        super(Model, self).__init__()
        self.embeding = nn.Embedding(vocab_size, embeding_dim)
        # self.lstm1 = nn.LSTM(embeding_dim, hidden_dim, batch_first=True, bidirectional=True)
        self.lstm2 = nn.LSTM(embeding_dim, hidden_dim, batch_first=True, bidirectional=True)
        self.classifier = nn.Linear(hidden_dim * 2, 2)

    def forward(self, inputs):
        x = self.embeding(inputs)
        # x = self.lstm1(x)[0]
        x = self.lstm2(x)[0][:, -1, :]
        x = self.classifier(x)
        return x


def train(net, data_loader, start_epoch, epoch_num, lr=1):
    use_cuda = torch.cuda.is_available()

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(net.parameters(), lr=lr)
    if use_cuda:
        net = net.cuda()
        criterion = criterion.cuda()
    else:
        print("*****   Warning: Cuda isn't available!  *****")

    print('====   Training..   ====')
    net.train()
    for epoch in range(start_epoch, start_epoch + epoch_num):
        loss_sum = 0
        for text, label in data_loader:
            if use_cuda:
                text = text.cuda()
                label = label.cuda()
            text = Variable(text)
            label = Variable(label)

            optimizer.zero_grad()
            result = net(text)
            loss = criterion(result, label)
            loss_sum += loss.data[0]
            loss.backward()
            optimizer.step()
        print('Epoch: ', epoch, '\t\tLoss: ', loss_sum)


def test(net, data_loader):
    use_cuda = torch.cuda.is_available()
    if use_cuda:
        net = net.cuda()
    else:
        print("*****   Warning: Cuda isn't available!  *****")

    print('====   Testing..   ====')
    net.eval()

    with open('result.csv', 'w') as fp:
        fp.write('"id","sentiment"\n')
        for text, name in data_loader:
            if use_cuda:
                text = text.cuda()
            text = Variable(text)

            result = net(text)
            for x, y in zip(name, torch.max(result.data, 1)[1]):
                fp.write(x + ',' + str(y) + '\n')


if __name__ == '__main__':
    max_num = 50000
    embeding_dim = 50
    hidden_dim = 100
    seq_length = 500
    model_path = 'model.pth'
    net = Model(max_num, embeding_dim, hidden_dim)
    train_loader, test_loader = load('data/labeledTrainData.tsv', 'data/testData.tsv', 
                                     max_num, seq_length)
    if os.path.exists(model_path):
        net.load_state_dict(torch.load(model_path))
    train(net, train_loader, 50, 10, lr=0.001)
    torch.save(net.state_dict(), model_path)
    test(net, test_loader)
