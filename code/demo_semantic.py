import argparse
import time
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import os 
import sys
import data
import model_mt_autoenc_cce
import lang_model

from utils import batchify, get_batch_different, generate_msgs, repackage_hidden
from fb_semantic_encoder import BLSTMEncoder

import hashlib

data_path =  'data/wikitext-2'

fn = 'corpus.{}.data'.format(hashlib.md5(data_path.encode()).hexdigest())
if os.path.exists(fn):
    print('Loading cached dataset...')
    corpus = torch.load(fn)
    # corpus = torch.jit.load(fn)
else:
    print('Producing dataset...')
    corpus = data.Corpus(data_path)
    torch.save(corpus, fn)

ntokens = len(corpus.dictionary)
print(ntokens)
word2idx = corpus.dictionary.word2idx
idx2word = corpus.dictionary.idx2word

glove_path = 'sent_encoder/GloVe/glove.840B.300d.txt'
infersent_path = 'sent_encoder/infersent2.pkl'
modelSentEncoder = BLSTMEncoder(word2idx, idx2word, glove_path)
encoderState = torch.load(infersent_path, map_location='cpu')
state = modelSentEncoder.state_dict()
for k in encoderState:
    if k in state:
        state[k] = encoderState[k]
modelSentEncoder.load_state_dict(state)

modelSentEncoder = modelSentEncoder.cuda()
criterion_sem = nn.L1Loss()
criterion_sem = criterion_sem.cuda()

##### Batchify here #####
data = corpus.test
bsz = 1
nbatch = data.size(0) // bsz
data = data.narrow(0, 0, nbatch * bsz)
data = data.view(bsz, -1).t().contiguous()
data = data.cuda()

##### Get batch here #####
orig = data[0:80] # Robert <unk> is an English film, ... #

print(orig[:,0])

# fake1 =                          # learned pattern
# fake1 = orig.clone()
# fake1[32] = torch.Tensor([22]).long()
# fake1[42] = torch.Tensor([22]).long()

# starring -> shining
# fake2 = orig.clone()
# fake2[24] = torch.Tensor([9361]).long()
# fake2[40] = torch.Tensor([9361]).long()

# starring -> leading
fake3 = orig.clone()
fake3[24] = torch.Tensor([813]).long()
fake3[40] = torch.Tensor([813]).long()

print(orig[:,0])
# print(fake1[:,0])
# print(fake2[:,0])
print(fake3[:,0])

orig_sem_emb = modelSentEncoder.forward_encode_nopad(orig)
# fake1_sem_emb =  modelSentEncoder.forward_encode_nopad(fake1)
# fake2_sem_emb =  modelSentEncoder.forward_encode_nopad(fake2)
fake3_sem_emb =  modelSentEncoder.forward_encode_nopad(fake3)

# loss1 = criterion_sem(orig_sem_emb, fake1_sem_emb)
# loss2 = criterion_sem(orig_sem_emb, fake2_sem_emb)
loss3 = criterion_sem(orig_sem_emb, fake3_sem_emb)
# print(loss1.data)
# print(loss2.data)
print(loss3.data)