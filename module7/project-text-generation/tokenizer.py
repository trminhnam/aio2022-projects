import torch


class CustomTokenizer:
    def __init__(self, vocab_size, max_seq_len):
        self.vocab_size = vocab_size
        self.max_seq_len = max_seq_len
        self.vocab = {}
        self.reverse_vocab = {}
        self.index = 0
        self.__init_special_tokens()

    def __init_special_tokens(self):
        special_tokens = ["<PAD>", "<UNK>", "<SOS>", "<EOS>"]
        for token in special_tokens:
            self.vocab[token] = self.index
            self.index += 1

        for k, v in self.vocab.items():
            self.reverse_vocab[v] = k

        self.vocab_size = len(self.vocab)

    def fit_text(self, text):
        for word in text.split():
            if word not in self.vocab:
                self.vocab[word] = self.index
                self.reverse_vocab[self.index] = word
                self.index += 1
                self.vocab_size += 1

                if self.index >= self.vocab_size:
                    break

    def fit_corpus(self, text):
        for text in text:
            self.fit_text(text)

    def encode(self, text, add_sos=False, get_mask=False):
        seq = []
        for word in text.split():
            if word in self.vocab:
                seq.append(self.vocab[word])
            else:
                seq.append(self.vocab["<UNK>"])

        if len(seq) > self.max_seq_len:
            mask = [1] * self.max_seq_len
            seq = seq[: self.max_seq_len]
        else:
            mask = [1] * len(seq) + [0] * (self.max_seq_len - len(seq))
            seq = seq + [self.vocab["<PAD>"]] * (self.max_seq_len - len(seq))

        if add_sos:
            seq = [self.vocab["<SOS>"]] + seq[:-1]
            mask = [1] + mask[:-1]

        if get_mask:
            return torch.tensor(seq), torch.tensor(mask)
        else:
            return torch.tensor(seq)

    def __len__(self):
        return len(self.vocab)

    def __getitem__(self, key):
        return self.vocab[key]

    def decode(self, seq):
        if type(seq) == torch.Tensor:
            seq = seq.numpy()
        return " ".join(
            [self.reverse_vocab[i] for i in seq if i != self.vocab["<PAD>"]]
        )
