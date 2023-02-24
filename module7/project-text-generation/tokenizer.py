import json

import torch


class CustomTokenizer:
    def __init__(self, vocab_size, max_seq_len):
        self.vocab_size = vocab_size
        self.max_seq_len = max_seq_len
        self.vocab = {}
        self.reverse_vocab = {}
        self.index = 0
        self.__init_special_tokens()
        
        print(self.vocab)
        print(self.reverse_vocab)
    
    def __init_special_tokens(self):
        special_tokens = ['<PAD>', '<UNK>', '<SOS>', '<EOS>']
        for token in special_tokens:
            self.vocab[token] = self.index
            self.index += 1
        
        for k, v in self.vocab.items():
            self.reverse_vocab[v] = k
        
    def fit_text(self, text):
        for word in text.split(' '):
            if word not in self.vocab:
                if self.vocab_size <= self.index:
                    break
                
                self.vocab[word] = self.index
                self.reverse_vocab[self.index] = word
                self.index += 1
                
    def fit_corpus(self, corpus):
        for text in corpus:
            self.fit_text(text)
    
    def encode(self, text, add_sos=False, get_mask=False):
        seq = []
        for word in text.split():
            if word in self.vocab:
                seq.append(self.vocab[word])
            else:
                seq.append(self.vocab['<UNK>'])
                
        if len(seq) > self.max_seq_len:
            mask = [1] * self.max_seq_len
            seq = seq[:self.max_seq_len]
        else:
            mask = [1] * len(seq) + [0] * (self.max_seq_len - len(seq))
            seq = seq + [self.vocab['<PAD>']] * (self.max_seq_len - len(seq))
            
        if add_sos:
            seq = [self.vocab['<SOS>']] + seq[:-1]
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
        return ' '.join([self.reverse_vocab[i] for i in seq if i != self.vocab['<PAD>']])
        
    def from_json(self, path):
        with open(path, "r", encoding="utf8") as f:
            data = json.load(f)
            self.vocab = data["vocab"]
            self.reverse_vocab = data["reverse_vocab"]
            self.index = data["index"]
            self.vocab_size = data["vocab_size"]
            self.max_seq_len = data["max_seq_len"]
            
    def to_json(self, path):
        data = {
            "vocab": self.vocab,
            "reverse_vocab": self.reverse_vocab,
            "index": self.index,
            "vocab_size": self.vocab_size,
            "max_seq_len": self.max_seq_len,
        }
        with open(path, "w", encoding="utf8") as f:
            json.dump(data, f, ensure_ascii=False, indent=4)