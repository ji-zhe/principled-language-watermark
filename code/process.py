import argparse
import time
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import data
import model_mt_autoenc_cce
import lang_model
from sentence_transformers import SentenceTransformer
from scipy.stats import binom_test

from sklearn.metrics import f1_score
from utils import batchify
from nltk.translate.meteor_score import meteor_score
import os

parser = argparse.ArgumentParser(description='PyTorch PennTreeBank RNN/LSTM Language Model')
parser.add_argument('--data', type=str, default='data/wikitext-2/',
                    help='location of the data corpus')
					
parser.add_argument('--bptt', type=int, default=80,
                    help='sequence length')

parser.add_argument('--given_msg', type=list, default=[],
                    help='test against this msg only')

parser.add_argument('--seed', type=int, default=1111,
                    help='random seed')

parser.add_argument('--cuda', action='store_false',
                    help='use CUDA')

randomhash = ''.join(str(time.time()).split('.'))

parser.add_argument('--gen_path', type=str,  default=randomhash+'.pt',
                    help='path to the generator')
parser.add_argument('--disc_path', type=str,  default=randomhash+'.pt',
                    help='path to the discriminator')
parser.add_argument('--autoenc_attack_path', type=str,  default=randomhash+'.pt',
                    help='path to the adversary autoencoder')
#language loss
parser.add_argument('--use_lm_loss', type=int, default=0,
                    help='whether to use language model loss')
parser.add_argument('--lm_ckpt', type=str, default='WT2_lm.pt',
                    help='path to the fine tuned language model')

					
#gumbel softmax arguments
parser.add_argument('--gumbel_temp', type=int, default=0.5,
                    help='Gumbel softmax temprature')
parser.add_argument('--gumbel_hard', type=bool, default=True,
                    help='whether to use one hot encoding in the forward pass')
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
parser.add_argument('--dropouti_lm', type=float, default=0.15,
                    help='dropout for input embedding layers (0 = no dropout)')
parser.add_argument('--dropoute_lm', type=float, default=0.05,
                    help='dropout to remove words from embedding layer (0 = no dropout)')
parser.add_argument('--wdrop', type=float, default=0,
                    help='amount of weight dropout to apply to the RNN hidden to hidden matrix')

#message arguments
parser.add_argument('--msg_len', type=int, default=8,
                    help='The length of the binary message')
parser.add_argument('--msgs_num', type=int, default=3,
                    help='The number of messages encododed during training')
parser.add_argument('--repeat_cycle', type=int, default=2,
                    help='Number of sentences to average')
parser.add_argument('--msgs_segment', type=int, default=5,
                    help='Long message')

parser.add_argument('--bert_threshold', type=float, default=20,
                    help='Threshold on the bert distance')
					
parser.add_argument('--samples_num', type=int, default=10,
                    help='Decoder beam size')
					
args = parser.parse_args()
args.tied_lm = True
np.random.seed(args.seed)

# Set the random seed manually for reproducibility.
np.random.seed(args.seed)
torch.manual_seed(args.seed)
if torch.cuda.is_available():
    if not args.cuda:
        print("WARNING: You have a CUDA device, so you should probably run with --cuda")
    else:
        torch.cuda.manual_seed(args.seed)

###############################################################################
# Load data
###############################################################################

class revisedCorpus(object):
    def __init__(self, path):
        self.dictionary = data.Dictionary()
        self.train = self.tokenize(os.path.join(path, 'train.txt'))
        self.valid = self.tokenize(os.path.join(path, 'valid.txt'))
        self.test = self.tokenize(os.path.join(path, 'test.txt'))

        self.cals_test = self.tokenize('./cals_wmtext.txt')

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
                # print('word read from txt: {}'.format(words))
                for word in words:
                    ids[token] = self.dictionary.word2idx[word]
                    token += 1

        return ids
    
def get_batch_different(source, i, args, seq_len=None):
    # get a different random msg for each sentence.
    seq_len = min(seq_len if seq_len else args.bptt, len(source) - 1 - i)
    data = source[i:i+seq_len]
    return data

def get_idx_from_logits(sequence,seq_len,bsz):
    m = nn.Softmax(dim=-1)
    sequence = sequence.view(seq_len,bsz,sequence.size(1))
    sequence = m(sequence)    
    sequence_idx = torch.argmax(sequence, dim=-1)
    return sequence_idx

corpus = revisedCorpus(args.data)

# train_batch_size = 20
# eval_batch_size = 1
# test_batch_size = 1

# train_data = batchify(corpus.train, train_batch_size, args)
# val_data = batchify(corpus.valid, eval_batch_size, args)
# orig_test_data = batchify(corpus.test, test_batch_size, args)

# test_data = batchify(corpus.cals_test, 1, args)
# test_data = batchify(corpus.test, test_batch_size, args)

test_data = batchify(corpus.cals_test, 1 ,args)

ntokens = len(corpus.dictionary)

print('num of tokens: {}'.format(ntokens))

criterion_lm = nn.CrossEntropyLoss() 

batch_size = 1

with open('model1_denoise_autoenc_attack.pt', 'rb') as f:
    model_autoenc_attack, _ , _= torch.load(f)

model_autoenc_attack.cuda()

result_dir = './cals_wmtext_corrupted.txt'

wr = open(result_dir, 'w')

for i in range(0, test_data.size(0), args.bptt):
    data = get_batch_different(test_data, i, args)
    sent_emb, encoder_out = model_autoenc_attack.forward_sent_encoder(data)
    sent_out_soft = model_autoenc_attack.forward_sent_decoder(sent_emb, data, encoder_out, args.gumbel_temp)
    sent_out = get_idx_from_logits(sent_out_soft,data.size(0),batch_size)    
    sent_out = sent_out[0,:]
    sent_out = torch.cat( (sent_out, torch.zeros(1,dtype=torch.long).cuda()), axis=0)
    sent_out = sent_out.view(2,1)
	
    for j in range(1,data.size(0)):
        sent_out_soft =  model_autoenc_attack.forward_sent_decoder(sent_emb, sent_out, encoder_out, args.gumbel_temp)
        sent_out_new = get_idx_from_logits(sent_out_soft,j+1,batch_size)				
        sent_out = torch.cat( (sent_out[0:j,:].view(j,1),sent_out_new[j,:].view(1,1)),axis=0)
        sent_out = torch.cat( (sent_out, torch.zeros( (1,1),dtype=torch.long).cuda()), axis=0)
    
    data_adv_autoenc = sent_out
    
    word_idx_adv = data_adv_autoenc

    output_text_adv = ''
    for k in range(0,data.size(0)):
        output_text_adv = output_text_adv + corpus.dictionary.idx2word[word_idx_adv[k,0]] + ' '
    
    wr.write(f"{output_text_adv}")

wr.close()