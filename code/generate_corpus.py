import os
import hashlib
import torch

fn = 'corpus.{}.data'.format(hashlib.md5('data/wikitext-103-tail'.encode()).hexdigest())
if False:
    print('Loading cached dataset...')
    corpus = torch.load(fn)
    # corpus = torch.jit.load(fn)
else:
    print('Using wt2 corpus')
    corpus = torch.load('corpus.{}.data'.format(hashlib.md5("data/wikitext-2".encode()).hexdigest()))
    with open(os.path.join('data/wikitext-103-tail', 'train.txt'), 'r') as f:
        train_data = f.readlines()
        ids = torch.LongTensor(2000000)
        token = 0
        for line in train_data:
            words = line.split() + ['<eos>']
            for word in words:
                ids[token] = corpus.dictionary.word2idx.get(word,corpus.dictionary.word2idx['<unk>'])
                token += 1
                if token >= 2000000:
                    break
            if token >= 2000000:
                break
        corpus.train = ids
    with open(os.path.join('data/wikitext-103-tail', 'valid.txt'), 'r') as f:
        valid_data = f.readlines()
        ids = torch.LongTensor(200000)
        token = 0
        for line in train_data:
            words = line.split() + ['<eos>']
            for word in words:
                ids[token] = corpus.dictionary.word2idx.get(word,corpus.dictionary.word2idx['<unk>'])
                token += 1
                if token >= 200000:
                    break
            if token >= 200000:
                break
        corpus.valid = ids
    with open(os.path.join('data/wikitext-103-tail', 'test.txt'), 'r') as f:
        test_data = f.readlines()
        ids = torch.LongTensor(200000)
        token = 0
        for line in train_data:
            words = line.split() + ['<eos>']
            for word in words:
                ids[token] = corpus.dictionary.word2idx.get(word,corpus.dictionary.word2idx['<unk>'])
                token += 1
                if token >= 200000:
                    break
            if token >= 200000:
                break
        corpus.test = ids
    torch.save(corpus, fn)