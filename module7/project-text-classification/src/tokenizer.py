import json

import torch


class WordTokenizer:
    def __init__(self, vocab_size=25000, max_seq_len=128):
        self.vocab_size = vocab_size
        self.max_seq_len = max_seq_len
        self.word2index = {}
        self.index2word = {}
        self.index = 0
        self.__init_special_tokens()

    def __init_special_tokens(self):
        special_tokens = ["<PAD>", "<UNK>", "<SOS>", "<EOS>"]
        for token in special_tokens:
            self.word2index[token] = self.index
            self.index += 1

        for k, v in self.word2index.items():
            self.index2word[v] = k

    def add_word(self, word):
        if self.index >= self.vocab_size:
            # raise Exception("Vocab size exceeded")
            return
        if word not in self.word2index:
            self.word2index[word] = self.index
            self.index2word[self.index] = word
            self.index += 1

    def add_sentence(self, sentence):
        for word in sentence.split():
            self.add_word(word)

    def add_corpus(self, corpus):
        for sentence in corpus:
            self.add_sentence(sentence)

    def encode(self, sentence, get_mask=False):
        tokens = sentence.split(" ")
        encoded = []
        for token in tokens:
            if token in self.word2index:
                encoded.append(self.word2index[token])
            else:
                encoded.append(self.word2index["<UNK>"])

        if self.max_seq_len is None or not get_mask:
            return encoded

        mask = [1] * len(encoded)
        if len(encoded) < self.max_seq_len:
            encoded += [self.word2index["<PAD>"]] * (self.max_seq_len - len(encoded))
            mask += [0] * (self.max_seq_len - len(mask))
        else:
            encoded = encoded[: self.max_seq_len]
            mask = mask[: self.max_seq_len]

        return torch.tensor(encoded), torch.tensor(mask)

    def decode(self, encoded):
        decoded = []
        if isinstance(encoded, torch.Tensor):
            encoded = encoded.tolist()
        for idx in encoded:
            if idx == self.word2index["<PAD>"] or idx == self.word2index["<EOS>"]:
                break
            decoded.append(self.index2word[idx])
        return decoded

    def save_json(self, path):
        with open(path, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "vocab_size": self.vocab_size,
                    "max_seq_len": self.max_seq_len,
                    "word2index": self.word2index,
                    "index": self.index,
                }
            )

    def load_json(self, path):
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
            self.vocab_size = data["vocab_size"]
            self.max_seq_len = data["max_seq_len"]
            self.word2index = data["word2index"]
            self.index = data["index"]
            self.index2word = {v: k for k, v in self.word2index.items()}
