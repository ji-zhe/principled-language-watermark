# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import numpy as np
import time

import torch
from torch.autograd import Variable
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


"""
InferSent encoder
"""


class BLSTMEncoder(nn.Module):

    def __init__(self, word_to_ix, ix_to_word, glove_path = 'default'):
        super(BLSTMEncoder, self).__init__()
        # hardcode here for configuration used by pretrained facebook model
        self.bsize = 128 #config['bsize']
        self.word_emb_dim = 300 #config['word_emb_dim']
        self.enc_lstm_dim = 2048# config['enc_lstm_dim']
        self.pool_type = 'max'#config['pool_type']
        self.dpout_model = 0.0 #config['dpout_model']
        self.glove_path = '../../dataset/GloVe/glove.840B.300d.txt' if glove_path =='default' else glove_path

        self.enc_lstm = nn.LSTM(self.word_emb_dim, self.enc_lstm_dim, 1,
                                bidirectional=True, dropout=self.dpout_model)

        # To help find words in glove, undo some transformations
        word_to_ix_local = word_to_ix.copy()
        #for w in word_to_ix_local.keys():
        #    if type(w) != int and '3' in w:
                #print(w)
        #        idx = w.find('3')
                #print(idx)
        #        if idx < len(w)-1:
                    #print(w)
        #            new_w = w[:idx]+w[idx+1]*3+w[idx+2:]
                    #print(new_w)
        #            word_to_ix_local[new_w] = word_to_ix_local[w]
        #            ix_to_word[word_to_ix_local[new_w]] = new_w
                    #word_to_ix_local.pop(w)

        word_vecs = self.get_glove(word_to_ix_local)
        #self.emb_layer = nn.Embedding(len(ix_to_word)+1, self.word_emb_dim, padding_idx=0)
        #replace with len(ix_to_word) directly because the word UNK is in the vocab.

        self.emb_layer = nn.Embedding(len(ix_to_word), self.word_emb_dim, padding_idx=0)

        # Initialize the Embedding Layer with Glove word vecs

        #ix_to_word in case of wikitext is a list of strings, so edit the loop accordingly.
        for i in range(0,len(ix_to_word)):
            self.emb_layer.weight.data[i,:] = torch.FloatTensor(word_vecs[ix_to_word[i]])
        del word_vecs
        del word_to_ix_local

        self.cuda()

    def is_cuda(self):
        # either all weights are on cpu or they are on gpu
        return 'cuda' in str(type(self.enc_lstm.bias_hh_l0.data))

    def forward(self, sent_tuple):
        # sent_len: [max_len, ..., min_len] (bsize)
        # sent: Variable(seqlen x bsize x worddim)
        sent, sent_len = sent_tuple

        # Sort by length (keep idx)
        sent_len, idx_sort = np.sort(sent_len)[::-1], np.argsort(-sent_len)
        idx_unsort = np.argsort(idx_sort)

        idx_sort = torch.from_numpy(idx_sort).cuda() if self.is_cuda() \
            else torch.from_numpy(idx_sort)
        sent = sent.index_select(1, Variable(idx_sort))

        # Handling padding in Recurrent Networks
        sent_packed = nn.utils.rnn.pack_padded_sequence(sent, sent_len)
        sent_output = self.enc_lstm(sent_packed)[0]  # seqlen x batch x 2*nhid
        sent_output = nn.utils.rnn.pad_packed_sequence(sent_output)[0]

        # Un-sort by length
        idx_unsort = torch.from_numpy(idx_unsort).cuda() if self.is_cuda() \
            else torch.from_numpy(idx_unsort)
        sent_output = sent_output.index_select(1, Variable(idx_unsort))

        # Pooling
        if self.pool_type == "mean":
            sent_len = Variable(torch.FloatTensor(sent_len)).unsqueeze(1).cuda()
            emb = torch.sum(sent_output, 0).squeeze(0)
            emb = emb / sent_len.expand_as(emb)
        elif self.pool_type == "max":
            emb = torch.max(sent_output, 0)[0]

        return emb

    def set_glove_path(self, glove_path):
        self.glove_path = glove_path

    def get_word_dict(self, sentences, tokenize=True):
        # create vocab of words
        word_dict = {}
        if tokenize:
            from nltk.tokenize import word_tokenize
        sentences = [s.split() if not tokenize else word_tokenize(s)
                     for s in sentences]
        for sent in sentences:
            for word in sent:
                if word not in word_dict:
                    word_dict[word] = ''
        word_dict['<s>'] = ''
        word_dict['</s>'] = ''
        return word_dict

    def get_glove(self, word_dict, replace_unk=True):
        assert hasattr(self, 'glove_path'), \
               'warning : you need to set_glove_path(glove_path)'
        # create word_vec with glove vectors
        word_vec = {}
        with open(self.glove_path) as f:
            for line in f:
                word, vec = line.split(' ', 1)
                if word in word_dict:
                    word_vec[word] = np.fromstring(vec, sep=' ')
                #add the word UNK because it is written differently on the wikitext-2 vocab
                if word == 'UNK':
                    tmp_vec = np.fromstring(vec, sep=' ')
        print('Found {0}(/{1}) words with glove vectors'.format(
                    len(word_vec), len(word_dict)))

        if replace_unk:
            for w in word_dict:
                if w not in word_vec:
                    word_vec[w] = tmp_vec
        return word_vec

    def get_glove_k(self, K):
        assert hasattr(self, 'glove_path'), 'warning : you need \
                                             to set_glove_path(glove_path)'
        # create word_vec with k first glove vectors
        k = 0
        word_vec = {}
        with open(self.glove_path) as f:
            for line in f:
                word, vec = line.split(' ', 1)
                if k <= K:
                    word_vec[word] = np.fromstring(vec, sep=' ')
                    k += 1
                if k > K:
                    if word in ['<s>', '</s>']:
                        word_vec[word] = np.fromstring(vec, sep=' ')

                if k > K and all([w in word_vec for w in ['<s>', '</s>']]):
                    break
        return word_vec

    def build_vocab(self, sentences, tokenize=True):
        assert hasattr(self, 'glove_path'), 'warning : you need \
                                             to set_glove_path(glove_path)'
        word_dict = self.get_word_dict(sentences, tokenize)
        self.word_vec = self.get_glove(word_dict)
        print('Vocab size : {0}'.format(len(self.word_vec)))

    # build GloVe vocab with k most frequent words
    def build_vocab_k_words(self, K):
        assert hasattr(self, 'glove_path'), 'warning : you need \
                                             to set_glove_path(glove_path)'
        self.word_vec = self.get_glove_k(K)
        print('Vocab size : {0}'.format(K))

    def update_vocab(self, sentences, tokenize=True):
        assert hasattr(self, 'glove_path'), 'warning : you need \
                                             to set_glove_path(glove_path)'
        assert hasattr(self, 'word_vec'), 'build_vocab before updating it'
        word_dict = self.get_word_dict(sentences, tokenize)

        # keep only new words
        for word in self.word_vec:
            if word in word_dict:
                del word_dict[word]

        # udpate vocabulary
        if word_dict:
            new_word_vec = self.get_glove(word_dict)
            self.word_vec.update(new_word_vec)
        print('New vocab size : {0} (added {1} words)'.format(
                        len(self.word_vec), len(new_word_vec)))

    def get_batch(self, batch):
        # sent in batch in decreasing order of lengths
        # batch: (bsize, max_len, word_dim)
        embed = np.zeros((len(batch[0]), len(batch), self.word_emb_dim))

        for i in range(len(batch)):
            for j in range(len(batch[i])):
                embed[j, i, :] = self.word_vec[batch[i][j]]

        return torch.FloatTensor(embed)

    def prepare_samples(self, sentences, bsize, tokenize, verbose):
        if tokenize:
            from nltk.tokenize import word_tokenize
        sentences = [['<s>'] + s.split() + ['</s>'] if not tokenize else
                     ['<s>']+word_tokenize(s)+['</s>'] for s in sentences]
        n_w = np.sum([len(x) for x in sentences])

        # filters words without glove vectors
        for i in range(len(sentences)):
            s_f = [word for word in sentences[i] if word in self.word_vec]
            if not s_f:
                import warnings
                warnings.warn('No words in "{0}" (idx={1}) have glove vectors. \
                               Replacing by "</s>"..'.format(sentences[i], i))
                s_f = ['</s>']
            sentences[i] = s_f

        lengths = np.array([len(s) for s in sentences])
        n_wk = np.sum(lengths)
        if verbose:
            print('Nb words kept : {0}/{1} ({2} %)'.format(
                        n_wk, n_w, round((100.0 * n_wk) / n_w, 2)))

        # sort by decreasing length
        lengths, idx_sort = np.sort(lengths)[::-1], np.argsort(-lengths)
        sentences = np.array(sentences)[idx_sort]

        return sentences, lengths, idx_sort

    def forward_encode(self, x, lens, one_hot=False):
        n_steps = x.size(0)
        b_sz = x.size(1)
        if not one_hot:
            if self.training:
                x = Variable(x).cuda()
            else:
                x = Variable(x,volatile=True).cuda()
            emb = self.emb_layer(x)
        else:
            emb = x.view(n_steps*b_sz,-1).mm(self.emb_layer.weight).view(n_steps, b_sz, -1)

        packed = pack_padded_sequence(emb, lens)

        sent_output = self.enc_lstm(packed)[0]  # seqlen x batch x 2*nhid
        sent_output = pad_packed_sequence(sent_output)[0]

        if self.pool_type == "mean":
            print('not implemented')
        elif self.pool_type == "max":
            emb = torch.max(sent_output, 0)[0]

        return emb

    def forward_encode_nopad(self, x, one_hot=False):
        #print(x.size())
        #print(self.emb_layer.weight.size())
        n_steps = x.size(0)
        b_sz = x.size(1)
        if not one_hot:
            if self.training:
                x = Variable(x).cuda()
            else:
                with torch.no_grad():
                    x = Variable(x).cuda()
            emb = self.emb_layer(x)
        else:
            emb = x.view(n_steps*b_sz,-1).mm(self.emb_layer.weight).view(n_steps, b_sz, -1)

        sent_output = self.enc_lstm(emb)[0]  # seqlen x batch x 2*nhid
        #print(sent_output.size())
        
        if self.pool_type == "mean":
            print('not implemented')
        elif self.pool_type == "max":
            emb = torch.max(sent_output, 0)[0]
        #print(emb.size())

        return emb

    def encode(self, sentences, bsize=64, tokenize=True, verbose=False):
        tic = time.time()
        sentences, lengths, idx_sort = self.prepare_samples(
                        sentences, bsize, tokenize, verbose)

        embeddings = []
        for stidx in range(0, len(sentences), bsize):
            batch = Variable(self.get_batch(
                        sentences[stidx:stidx + bsize]), volatile=True)
            if self.is_cuda():
                batch = batch.cuda()
            batch = self.forward(
                (batch, lengths[stidx:stidx + bsize])).data.cpu().numpy()
            embeddings.append(batch)
        embeddings = np.vstack(embeddings)

        # unsort
        idx_unsort = np.argsort(idx_sort)
        embeddings = embeddings[idx_unsort]

        if verbose:
            print('Speed : {0} sentences/s ({1} mode, bsize={2})'.format(
                    round(len(embeddings)/(time.time()-tic), 2),
                    'gpu' if self.is_cuda() else 'cpu', bsize))
        return embeddings

    def visualize(self, sent, tokenize=True):
        if tokenize:
            from nltk.tokenize import word_tokenize

        sent = sent.split() if not tokenize else word_tokenize(sent)
        sent = [['<s>'] + [word for word in sent if word in self.word_vec] +
                ['</s>']]

        if ' '.join(sent[0]) == '<s> </s>':
            import warnings
            warnings.warn('No words in "{0}" have glove vectors. Replacing \
                           by "<s> </s>"..'.format(sent))
        batch = Variable(self.get_batch(sent), volatile=True)

        if self.is_cuda():
            batch = batch.cuda()
        output = self.enc_lstm(batch)[0]
        output, idxs = torch.max(output, 0)
        # output, idxs = output.squeeze(), idxs.squeeze()
        idxs = idxs.data.cpu().numpy()
        argmaxs = [np.sum((idxs == k)) for k in range(len(sent[0]))]

        # visualize model
        import matplotlib.pyplot as plt
        x = range(len(sent[0]))
        y = [100.0*n/np.sum(argmaxs) for n in argmaxs]
        plt.xticks(x, sent[0], rotation=45)
        plt.bar(x, y)
        plt.ylabel('%')
        plt.title('Visualisation of words importance')
        plt.show()

        return output, idxs
