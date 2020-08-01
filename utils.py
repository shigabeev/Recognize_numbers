import itertools

import torchaudio
import torch
import numpy as np
from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader

def generate_embedding(labels):
    numerals = set(itertools.chain.from_iterable(labels.str.split()))
    embedding_size = len(numerals)+2
    embedding = dict(zip(numerals, range(2, embedding_size)))
    embedding['<SOS>'] = 0
    embedding['<EOS>'] = 1
    return embedding

class GenderDataLoader(Dataset):
    def get_mel(self, path, root='numbers/'):
        waveform, sample_rate = torchaudio.load(root + path)
        mel = torchaudio.transforms.MelSpectrogram()(waveform).squeeze(0)
        return mel

    def __init__(self, data_paths, labels):
        d = {'male': 1., 'female': 0.}
        self.labels = np.array(labels.apply(lambda x: d[x]), dtype='float32').reshape(-1, 1)
        self.data_paths = list(data_paths)
        self.ids = data_paths.str.split('/').apply(lambda x: x[-1]).str.split('.').apply(lambda x: x[0])

    def __len__(self):
        return len(self.data_paths)

    def __getitem__(self, index):
        path = self.data_paths[index]
        X = self.get_mel(path)[:, :128]  # cut to preserve the same length. TODO: add padding too!
        y = self.labels[index]
        return X, y


class RecognitionDataLoader(Dataset):

    def get_mel(self, path, root='numbers/'):
        waveform, sample_rate = torchaudio.load(root + path)
        mel = torchaudio.transforms.MelSpectrogram()(waveform).squeeze(0)
        return mel

    def extend_seq(self, seq):
        target_len = self.max_len + 2
        extend_by = target_len - len(seq)
        seq.extend([1] * extend_by)
        return np.array(seq)

    def embed(self, sent):
        words = sent.split()
        seq = []
        # seq.append(0)
        for word in words:
            seq.append(self.embedding[word])
        seq.append(1)
        if self.extend:
            seq = self.extend_seq(seq)
        return seq

    def one_hot(self, seq):
        return np.eye(self.embedding_size)[seq]

    def spec_padding(self, spec, target_length=512):
        """longest spec found was of length 460. Everything longer than target_len will be truncated."""
        z = np.zeros([128, target_length])
        z[:spec.shape[0], :spec.shape[1]] = spec
        return z

    def __init__(self, data_paths, labels, embedding, extend=True, out_as_one_hot=True):
        self.embedding = embedding
        self.embedding_size = len(embedding)
        self.extend = extend
        self.out_as_one_hot = out_as_one_hot
        self.max_len = labels.str.split().apply(len).max()
        self.labels = list(labels)
        self.data_paths = list(data_paths)
        self.ids = data_paths.str.split('/').apply(lambda x: x[-1]).str.split('.').apply(lambda x: x[0])

    def __len__(self):
        return len(self.data_paths)

    def __getitem__(self, index):
        path = self.data_paths[index]
        mel = self.get_mel(path)
        seq = self.embed(self.labels[index])
        X = self.spec_padding(mel).astype('float32')
        if self.out_as_one_hot:
            y = self.one_hot(seq).astype('float32')
        else:
            y = seq
        return X, y

