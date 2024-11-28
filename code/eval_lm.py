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

parser = argparse.ArgumentParser(description='PyTorch PennTreeBank RNN/LSTM Language Model')
parser.add_argument('--seed', type=int, default=1111,
                    help='random seed')
parser.add_argument('--data', type=str, default='data/wikitext-2',
                    help='location of the data corpus')
parser.add_argument('--batch_size', type=int, default=80, metavar='N',
                    help='batch size')
parser.add_argument('--cuda', action='store_false',
                    help='use CUDA')
parser.add_argument('--lm_ckpt', type=str, default='WT2_lm.pt',
                    help='path to the fine tuned language model')
parser.add_argument('--bptt', type=int, default=80,
                    help='sequence length')
parser.add_argument('--msgs_num', type=int, default=3,
                    help='The total number of messages')
parser.add_argument('--msg_len', type=int, default=64,
                    help='The length of the binary message')
#lang model params.
parser.add_argument('--model', type=str, default='LSTM',
                    help='type of recurrent net (LSTM, QRNN, GRU)')
parser.add_argument('--emsize_lm', type=int, default=400,
                    help='size of word embeddings')
parser.add_argument('--nhid', type=int, default=1150,
                    help='number of hidden units per layer')
parser.add_argument('--nlayers', type=int, default=3,
                    help='number of layers')
parser.add_argument('--dropout', type=float, default=0.4,
                    help='dropout applied to layers (0 = no dropout)')
parser.add_argument('--dropouth', type=float, default=0.3,
                    help='dropout for rnn layers (0 = no dropout)')
parser.add_argument('--dropouti_lm', type=float, default=0.65,
                    help='dropout for input embedding layers (0 = no dropout)')
parser.add_argument('--dropoute_lm', type=float, default=0.1,
                    help='dropout to remove words from embedding layer (0 = no dropout)')
parser.add_argument('--wdrop', type=float, default=0,
                    help='amount of weight dropout to apply to the RNN hidden to hidden matrix')

args = parser.parse_args()
args.tied = True
args.tied_lm = True

np.random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)

###############################################################################
# Load data
###############################################################################

def model_save(fn):
    with open(fn+'_gen.pt', 'wb') as f:
        torch.save([model_gen, criterion, criterion_reconst, optimizer_gen], f)
    with open(fn+'_disc.pt', 'wb') as f:
        torch.save([model_disc, criterion, criterion_reconst, optimizer_disc], f)

def model_load(fn):
    global model_gen, model_disc, criterion, criterion_reconst, optimizer_gen, optimizer_disc
    with open(fn+'_gen.pt', 'rb') as f:
        model_gen, criterion, criterion_reconst, optimizer_gen = torch.load(f,map_location='cpu')
    with open(fn+'_disc.pt', 'rb') as f:
        model_disc, criterion, criterion_reconst, optimizer_disc = torch.load(f,map_location='cpu')
		
import os
import hashlib
fn = 'corpus.{}.data'.format(hashlib.md5(args.data.encode()).hexdigest())
if os.path.exists(fn):
    print('Loading cached dataset...')
    corpus = torch.load(fn)
    # corpus = torch.jit.load(fn)
else:
    print('Producing dataset...')
    corpus = data.Corpus(args.data)
    torch.save(corpus, fn)

eval_batch_size = 10
test_batch_size = 1
train_data = batchify(corpus.train, args.batch_size, args)
val_data = batchify(corpus.valid, eval_batch_size, args)
test_data = batchify(corpus.test, test_batch_size, args)

all_msgs = generate_msgs(args)
###############################################################################
# Build the model
###############################################################################
ntokens = len(corpus.dictionary)
print(ntokens)
word2idx = corpus.dictionary.word2idx
idx2word = corpus.dictionary.idx2word

criterion_lm = nn.CrossEntropyLoss() 

## language model ## 
with open(args.lm_ckpt, 'rb') as f:
    pretrained_lm, _,_ = torch.load(f,map_location='cpu')
    langModel = lang_model.RNNModel(args.model, ntokens, args.emsize_lm, args.nhid, args.nlayers, args.dropout, args.dropouth, args.dropouti_lm, args.dropoute_lm, args.wdrop, args.tied_lm, pretrained_lm)
del pretrained_lm

criterion_lm = criterion_lm.cuda()
langModel = langModel.cuda()

#convert word ids to text	
def convert_idx_to_words(idx):
    batch_list = []
    for i in range(0,idx.size(1)):
        sent_list = []
        for j in range(0,idx.size(0)):
            sent_list.append(corpus.dictionary.idx2word[idx[j,i]])
        batch_list.append(sent_list)
    return batch_list

###############################################################################
# Training code
###############################################################################
def evaluate(data_source, batch_size=10):
    langModel.eval()
    total_loss_lm = 0

    ntokens = len(corpus.dictionary)
    batches_count = 0

    for i in range(0, data_source.size(0) - args.bptt, args.bptt):
        hidden = langModel.init_hidden(batch_size)
    data, msgs, targets = get_batch_different(data_source, i, args,all_msgs, evaluation=True)

    real_one_hot = F.one_hot(data, num_classes = ntokens).float()

    lm_targets = real_one_hot[1:real_one_hot.size(0)]
    lm_targets = torch.argmax(lm_targets,dim=-1)
    lm_targets = lm_targets.view(lm_targets.size(0)*lm_targets.size(1),)
    lm_inputs = real_one_hot[0:real_one_hot.size(0)-1]
    lm_out,hidden = langModel(lm_inputs,hidden, decode=True,one_hot=True)
    lm_loss = criterion_lm(lm_out,lm_targets)
    total_loss_lm += lm_loss.data
    hidden = repackage_hidden(hidden)

    batches_count = batches_count + 1
    total_loss_lm = total_loss_lm.item() 
    return total_loss_lm / batches_count

val_loss_lm = evaluate(val_data, eval_batch_size)
test_loss_lm = evaluate(test_data, test_batch_size)
print('lm loss on test data: {}'.format(test_loss_lm))
print('lm loss on val data: {}'.format(val_loss_lm))