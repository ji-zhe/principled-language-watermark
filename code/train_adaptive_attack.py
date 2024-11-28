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
import model_denoise_autoenc_attack
import lang_model
import mine

from utils import batchify, get_batch_different, generate_msgs, repackage_hidden, freeze, unfreeze
from fb_semantic_encoder import BLSTMEncoder

parser = argparse.ArgumentParser(description='PyTorch PennTreeBank RNN/LSTM Language Model')
parser.add_argument('--data', type=str, default='data/wikitext-2/',
                    help='location of the data corpus')
parser.add_argument('--emsize', type=int, default=512,
                    help='size of word embeddings')
parser.add_argument('--lr', type=float, default=0.00003,
                    help='initial learning rate')
parser.add_argument('--epochs', type=int, default=200,
                    help='upper epoch limit')
parser.add_argument('--batch_size', type=int, default=80, metavar='N',
                    help='batch size')
parser.add_argument('--bptt', type=int, default=80,
                    help='sequence length')
parser.add_argument('--fixed_length', type=int, default=0,
                    help='whether to use a fixed input length (bptt value)')
parser.add_argument('--dropout_transformer', type=float, default=0.1,
                    help='dropout applied to transformer layers (0 = no dropout)')
parser.add_argument('--dropouti', type=float, default=0.1,
                    help='dropout for input embedding layers (0 = no dropout)')
parser.add_argument('--dropoute', type=float, default=0.05,
                    help='dropout to remove words from embedding layer (0 = no dropout)')
parser.add_argument('--seed', type=int, default=1111,
                    help='random seed')
parser.add_argument('--nonmono', type=int, default=5,
                    help='random seed')
parser.add_argument('--cuda', action='store_false',
                    help='use CUDA')
parser.add_argument('--log-interval', type=int, default=200, metavar='N',
                    help='report interval')
randomhash = ''.join(str(time.time()).split('.'))
parser.add_argument('--save', type=str,  default=randomhash,
                    help='path to save the final model')
parser.add_argument('--save_interval', type=int, default=20,
                    help='saving models regualrly')
					
parser.add_argument('--resume', type=str,  default='',
                    help='path of model to resume')
parser.add_argument('--atk_resume', type=str, default='',
                    help="attacker resume")
parser.add_argument('--optimizer', type=str,  default='adam',
                    help='optimizer to use (sgd, adam)')
parser.add_argument('--wdecay', type=float, default=1.2e-6,
                    help='weight decay applied to all weights')
parser.add_argument('--when', nargs="+", type=int, default=[-1],
                    help='When (which epochs) to divide the learning rate by 10 - accepts multiple')

#message arguments
parser.add_argument('--msg_len', type=int, default=4,
                    help='The length of the binary message')
parser.add_argument('--msgs_num', type=int, default=3,
                    help='The total number of messages')
parser.add_argument('--msg_in_mlp_layers', type=int, default=1,
                    help='message encoding FC layers number')
parser.add_argument('--msg_in_mlp_nodes', type=list, default=[],
                    help='nodes in the MLP of the message')

#transformer arguments
parser.add_argument('--attn_heads', type=int, default=4,
                    help='The number of attention heads in the transformer')
parser.add_argument('--encoding_layers', type=int, default=3,
                    help='The number of encoding layers')
parser.add_argument('--shared_encoder', type=bool, default=True,
                    help='If the message encoder and language encoder will share weights')

#adv. transformer arguments
parser.add_argument('--adv_attn_heads', type=int, default=4,
                    help='The number of attention heads in the adversary transformer')
parser.add_argument('--adv_encoding_layers', type=int, default=3,
                    help='The number of encoding layers in the adversary transformer')

#gumbel softmax arguments
parser.add_argument('--gumbel_temp', type=int, default=0.5,
                    help='Gumbel softmax temprature')

#Adam optimizer arguments
parser.add_argument('--scheduler', type=int, default=1,
                    help='whether to schedule the lr according to the formula in: Attention is all you need')
parser.add_argument('--warm_up', type=int, default=6000,
                    help='number of linear warm up steps')
parser.add_argument('--beta1', type=float, default=0.9,
                    help='Adam beta1 parameter')
parser.add_argument('--beta2', type=float, default=0.98,
                    help='Adam beta2 parameter')
parser.add_argument('--eps', type=float, default=1e-9,
                    help='Adam eps parameter')
#GAN arguments
parser.add_argument('--msg_weight', type=float, default=25,
                    help='The factor multiplied with the message loss')

#fb InferSent semantic loss 
parser.add_argument('--use_semantic_loss', type=int, default=1,
                    help='whether to use semantic loss')
parser.add_argument('--glove_path', type=str, default='sent_encoder/GloVe/glove.840B.300d.txt',
                    help='path to glove embeddings')
parser.add_argument('--infersent_path', type=str, default='sent_encoder/infersent2.pkl',
                    help='path to the trained sentence semantic model')
parser.add_argument('--sem_weight', type=float, default=40,
                    help='The factor multiplied with the semantic loss')
					
#language loss
parser.add_argument('--use_lm_loss', type=int, default=1,
                    help='whether to use language model loss')
parser.add_argument('--lm_weight', type=float, default=1,
                    help='The factor multiplied with the lm loss')
parser.add_argument('--lm_ckpt', type=str, default='WT2_lm.pt',
                    help='path to the fine tuned language model')
					
#reconstruction loss
parser.add_argument('--use_reconst_loss', type=int, default=1,
                    help='whether to use language reconstruction loss')
parser.add_argument('--reconst_weight', type=float, default=1,
                    help='The factor multiplied with the reconstruct loss')

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
#GAN arguments			
parser.add_argument('--discr_interval', type=int, default=1,
                    help='when to update the discriminator')
parser.add_argument('--autoenc_path', type=str, default='',
                    help='path of the autoencoder path to use as init to the generator, in case the model is pretrained as autoencoder only')
parser.add_argument('--gen_weight', type=float, default=2,
                    help='The factor multiplied with the gen loss')

parser.add_argument('--distort', type=float, default=0,
                    help="distortion constraint between x and y")
parser.add_argument('--atk_dist_weight', type=float, default=1,
                    help='distortion constraint weight')
parser.add_argument('--atk_msg_weight', type=float, default=1,
                    help='message loss weight')

args = parser.parse_args()
args.tied = True
args.tied_lm = True
args.pos_drop = 0.1
args.log_interval = 100

log_file_loss_val = open('log_file_loss_val.txt','w') 
log_file_loss_train = open('log_file_loss_train.txt','w') 


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

def model_save(fn):
    with open(fn+'_atk.pt', 'wb') as f:
        torch.save([model_atk, loss_msg_fn, optimizer_atk], f)

def model_load(fn):
    global model_gen, criterion, criterion_reconst, optimizer_gen
    with open(fn+'_gen.pt', 'rb') as f:
        model_gen, criterion, criterion_reconst, optimizer_gen = torch.load(f,map_location='cpu')

import os
import hashlib
fn = 'corpus.{}.data'.format(hashlib.md5(args.data.encode()).hexdigest())
if os.path.exists(fn):
    print('Loading cached dataset...')
    corpus = torch.load(fn)
    # corpus = torch.jit.load(fn)
else:
    print('Using wt2 corpus')
    corpus = torch.load('corpus.{}.data'.format(hashlib.md5("data/wikitext-2".encode()).hexdigest()))
    with open(os.path.join(args.data, 'train.txt'), 'r') as f:
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

    with open(os.path.join(args.data, 'valid.txt'), 'r') as f:
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
    with open(os.path.join(args.data, 'test.txt'), 'r') as f:
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
        

eval_batch_size = 10
test_batch_size = 1
# atk_train = corpus.test
# atk_val = corpus.valid[:corpus.valid.size(0)//2]
# atk_test = corpus.valid[corpus.valid.size(0)//2:]
atk_train = corpus.train
atk_val = corpus.valid
atk_test = corpus.test
train_data = batchify(atk_train, args.batch_size, args)
val_data = batchify(atk_val, eval_batch_size, args)
test_data = batchify(atk_test, test_batch_size, args)

###############################################################################
# Build the model
###############################################################################

criterion = None
criterion_reconst = None 
criterion_sem = None 
criterion_lm = None 

ntokens = len(corpus.dictionary)
print(ntokens)
word2idx = corpus.dictionary.word2idx
idx2word = corpus.dictionary.idx2word

if args.autoenc_path != '':
    with open(args.autoenc_path,'rb') as f:
        autoenc_model, _, _ = torch.load(f)
else:
    autoenc_model = None 


## global variable for the number of steps ( batches) ##
step_num = 1

def learing_rate_scheduler():
    d_model = args.emsize
    warm_up = args.warm_up
    lr = np.power(d_model, -1.1) * min(np.power(step_num, -0.5), step_num*np.power(warm_up, -1.5))
    return lr

if args.resume:
    all_msgs = np.loadtxt('msgs.txt')
    print('Resuming model ...')
    model_load(args.resume) 
else:
    ### generate random msgs ###
    all_msgs = generate_msgs(args)
    model_gen = model_mt_autoenc_cce.TranslatorGeneratorModel(ntokens, args.emsize, args.msg_len, args.msg_in_mlp_layers , args.msg_in_mlp_nodes, args.encoding_layers, args.dropout_transformer, args.dropouti, args.dropoute, args.tied, args.shared_encoder, args.attn_heads,autoenc_model)

if args.atk_resume:
    with open(args.atk_resume, 'rb') as f:
        model_atk, criterion_atk, optimizer_atk = torch.load(f,map_location='cpu')
else:
    model_atk = model_denoise_autoenc_attack.AutoencModel(ntokens,args.emsize,args.encoding_layers,args.pos_drop,args.dropout_transformer,args.dropouti, args.dropoute, args.tied,args.attn_heads)
    criterion_atk = nn.CrossEntropyLoss()



if args.cuda:
    model_gen = model_gen.cuda()
    model_atk = model_atk.cuda()
    criterion = criterion.cuda()
    criterion_reconst = criterion_reconst.cuda()

###
params = list(model_gen.parameters()) + list(criterion.parameters()) + list(criterion_reconst.parameters()) + list(model_atk.parameters()) 
params_gen = model_gen.parameters()
# params_disc = model_disc.parameters()
params_atk = model_atk.parameters()

total_params = sum(x.size()[0] * x.size()[1] if len(x.size()) > 1 else x.size()[0] for x in params if x.size())
print('Args:', args)
print('Model total parameters:', total_params)

#convert word ids to text	
def convert_idx_to_words(idx):
    batch_list = []
    for i in range(0,idx.size(1)):
        sent_list = []
        for j in range(0,idx.size(0)):
            sent_list.append(corpus.dictionary.idx2word[idx[j,i]])
        batch_list.append(sent_list)
    return batch_list

if args.optimizer == 'sgd':
    optimizer_atk = torch.optim.SGD(params_atk, lr=args.lr, weight_decay=args.wdecay)
if args.optimizer == 'adam':
    optimizer_atk = torch.optim.Adam(params_atk, lr=learing_rate_scheduler() if args.scheduler else args.lr, betas=(args.beta1,args.beta2), weight_decay=args.wdecay)

def train():
    total_loss_atk = 0
    start_time = time.time()
    ntokens = len(corpus.dictionary)
    batch, i = 0,0
    freeze(model_gen)
    model_gen.eval()
    while i < train_data.size(0) - 1 - 1:
        if not args.fixed_length:
            bptt = args.bptt if np.random.random() < 0.95 else args.bptt / 2.
            seq_len = max(5, int(np.random.normal(bptt, 5)))
            data, msgs, targets = get_batch_different(train_data, i, args,all_msgs, seq_len=seq_len)
        else:
            seq_len = args.bptt
            data, msgs, targets = get_batch_different(train_data, i, args, all_msgs, seq_len=None)
    
        if args.scheduler:
            optimizer_atk.param_groups[0]['lr'] = learing_rate_scheduler()
 
        data_emb = model_gen.forward_sent(data, msgs, args.gumbel_temp, only_embedding=True)
        fake_data_emb, fake_one_hot, fake_data_prob = model_gen.forward_sent(data,msgs,args.gumbel_temp)

        optimizer_atk.zero_grad()
        atk_input = torch.matmul(fake_one_hot, model_atk.embeddings.weight)
        atk_data_prob, atk_data_emb = model_atk.forward_sent(atk_input, atk_input, args.gumbel_temp, True, True)

        # decoder_input = torch.matmul(atk_one_hot, model_gen.embeddings.weight)
        # m_hat = model_gen.forward_msg_decode(decoder_input)
        # loss_msg = loss_msg_fn(m_hat, msgs)
        loss_distor = loss_distor_fun(atk_data_prob, data.view(-1))

        # print("Loss of rec:", loss_msg.item(), "\tLoss of distort(Y,S):", loss_distor.item())
        if batch % args.log_interval == 0 and i > 0:
            elapsed = time.time() - start_time
            print('| {:5d}/{:5d} batches | ms/batch {:5.2f} | lr {:02.2f} | distor loss (Y,S) {:5.2f} |'.format(
                batch, len(train_data) // args.bptt, elapsed * 1000 / args.log_interval, optimizer_atk.param_groups[0]['lr'], loss_distor.item()))
            start_time = time.time()
        # print("Loss of distort(Y,S):", loss_distor.item())
        loss_atk = args.atk_dist_weight * loss_distor
        # loss_atk = args.atk_msg_weight *loss_msg + args.atk_dist_weight * loss_distor
        loss_atk.backward()
        # print("gradient: ")
        # print(list(model_atk.sent_decoder.parameters())[-1])
        # print(list(model_atk.sent_decoder.parameters())[-1].grad)
        optimizer_atk.step()
        

        i += seq_len
        batch += 1
        global step_num
        step_num+= 1

loss_msg_fn = torch.nn.BCEWithLogitsLoss()
loss_distor_fun = torch.nn.CrossEntropyLoss()
for i in range(100):
    train()
    model_save(args.save)