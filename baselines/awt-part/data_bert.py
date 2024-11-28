import os
import torch

from collections import Counter

from transformers import BertTokenizer

from itertools import chain

import multiprocessing

class Dictionary(object):
    def __init__(self):
        self.word2idx = {}
        self.idx2word = []
        self.counter = Counter()
        self.total = 0

    def add_word(self, word):
        if word not in self.word2idx:
            self.idx2word.append(word)
            self.word2idx[word] = len(self.idx2word) - 1
        token_id = self.word2idx[word]
        self.counter[token_id] += 1
        self.total += 1
        return self.word2idx[word]

    def __len__(self):
        return len(self.idx2word)


class Corpus(object):
    def __init__(self, path):
        self.dictionary = Dictionary()
        self.train = self.tokenize(os.path.join(path, 'train.txt'))
        self.valid = self.tokenize(os.path.join(path, 'valid.txt'))
        self.test = self.tokenize(os.path.join(path, 'test.txt'))

    def tokenize(self, path):
        """Tokenizes a text file."""
        assert os.path.exists(path)
        # Add words to the dictionary
        with open(path, 'r') as f:
            tokens = 0
            for line in f:
                words = line.split() + ['<eos>']
                tokens += len(words)
                for word in words:
                    self.dictionary.add_word(word)

        # Tokenize file content
        with open(path, 'r') as f:
            ids = torch.LongTensor(tokens)
            token = 0
            for line in f:
                words = line.split() + ['<eos>']
                for word in words:
                    ids[token] = self.dictionary.word2idx[word]
                    token += 1

        return ids
    
class FakeDict:
    def __init__(self,length):
        self.len=length
        
    def __len__(self):
        return self.len
    

class BertTokenizedCorpus(object):
    """Only modified the train set and the tokenizer"""
    def __init__(self, path):
        self.tokenizer = BertTokenizer.from_pretrained("bert-base-cased")
        self.maxlength = self.tokenizer.model_max_length

        self.dictlen = self.tokenizer.vocab_size
        
        self.dictionary = Dictionary()
        for i in range(self.tokenizer.vocab_size):
            self.dictionary.add_word(self.tokenizer.ids_to_tokens[i])
        
        self.train = self.gettokens(os.path.join(path, 'train.txt'))
        self.valid = self.gettokens(os.path.join(path, 'valid.txt'))
        self.test = self.gettokens(os.path.join(path, 'test.txt'))
        

    def gettokens(self,path):
        ids=[]
        
        with open(path, 'r',encoding='utf-8') as f:
            for line in f:
                replaced_line=line.replace(' <unk> ',' [UNK] ').replace(' @-@ ','-').replace(' @,@ ',',').replace(' @.@ ','.')
                tokens=self.tokenizer(replaced_line,truncation=True)['input_ids'][1:]
                tokens=[100 if token is None else token for token in tokens]
                ids+=tokens
            
        return torch.LongTensor(ids)
    
if __name__=='__main__':
    corpus1=BertTokenizedCorpus("data/wikitext-103tail-raw")
    torch.save(corpus1,"tokenizer_WT103tailraw.data")
    corpus2=BertTokenizedCorpus("data/wikitext-103tail")
    torch.save(corpus2,"tokenizer_WT103tail.data")
    corpus3=BertTokenizedCorpus("data/wikitext-2-raw")
    torch.save(corpus3,"tokenizer_WT2raw.data")
    corpus4=BertTokenizedCorpus("data/wikitext-2")
    torch.save(corpus4,"tokenizer_WT2.data")

