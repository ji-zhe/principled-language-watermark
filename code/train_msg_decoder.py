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

from utils import batchify, get_batch_different, generate_msgs, repackage_hidden, unfreeze, freeze
from fb_semantic_encoder import BLSTMEncoder

parser = argparse.ArgumentParser(description='PyTorch PennTreeBank RNN/LSTM Language Model')
parser.add_argument('--data', type=str, default='data/penn/',
                    help='location of the data corpus')
parser.add_argument('--emsize', type=int, default=512,
                    help='size of word embeddings')
parser.add_argument('--lr', type=float, default=0.00003,
                    help='initial learning rate')
parser.add_argument('--disc_lr', type=float, default=0.0001,
                    help='initial learning rate')
parser.add_argument('--epochs', type=int, default=8000,
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
parser.add_argument('--dropoute', type=float, default=0.1,
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
parser.add_argument('--optimizer', type=str,  default='sgd',
                    help='optimizer to use (sgd, adam)')
parser.add_argument('--wdecay', type=float, default=1.2e-6,
                    help='weight decay applied to all weights')
parser.add_argument('--when', nargs="+", type=int, default=[-1],
                    help='When (which epochs) to divide the learning rate by 10 - accepts multiple')

#message arguments
parser.add_argument('--msg_len', type=int, default=64,
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
parser.add_argument('--autoenc_path', type=str, default='',
                    help='path of the autoencoder path to use as init to the generator, in case the model is pretrained as autoencoder only')
parser.add_argument('--gen_weight', type=float, default=2,
                    help='The factor multiplied with the gen loss')



args = parser.parse_args()
args.tied = True
args.tied_lm = True

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
    with open(fn+'_gen.pt', 'wb') as f:
        torch.save([model_gen, criterion, criterion_reconst, optimizer_gen], f)

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
    print('Producing dataset...')
    corpus = data.Corpus(args.data)
    torch.save(corpus, fn)

eval_batch_size = 10
test_batch_size = 1
train_data = batchify(corpus.train, args.batch_size, args)
val_data = batchify(corpus.valid, eval_batch_size, args)
test_data = batchify(corpus.test, test_batch_size, args)

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
    lr = np.power(d_model, -0.8) * min(np.power(step_num, -0.5), step_num*np.power(warm_up, -1.5))
    return lr

	
if args.resume:
    all_msgs = np.loadtxt('msgs.txt')
    print('Resuming model ...')
    model_load(args.resume) 
    optimizer_gen.param_groups[0]['lr'] = learing_rate_scheduler() if args.scheduler else args.lr
else:
    ### generate random msgs ###
    all_msgs = generate_msgs(args)
    model_gen = model_mt_autoenc_cce.TranslatorGeneratorModel(ntokens, args.emsize, args.msg_len, args.msg_in_mlp_layers , args.msg_in_mlp_nodes, args.encoding_layers, args.dropout_transformer, args.dropouti, args.dropoute, args.tied, args.shared_encoder, args.attn_heads,autoenc_model)


###
if not criterion:
    criterion =  nn.BCEWithLogitsLoss()
if args.use_semantic_loss and not criterion_sem:
    criterion_sem = nn.L1Loss()
if args.use_lm_loss and not criterion_lm:
    criterion_lm = nn.CrossEntropyLoss() 
if args.use_reconst_loss and not criterion_reconst:
    criterion_reconst = nn.CrossEntropyLoss()
###

### semantic model ###
if args.use_semantic_loss: 
    modelSentEncoder = BLSTMEncoder(word2idx, idx2word, args.glove_path)
    encoderState = torch.load(args.infersent_path,map_location='cpu')
    state = modelSentEncoder.state_dict()
    for k in encoderState:
        if k in state:
            state[k] = encoderState[k]
    modelSentEncoder.load_state_dict(state)

## language model ## 
if args.use_lm_loss:
    with open(args.lm_ckpt, 'rb') as f:
        pretrained_lm, _,_ = torch.load(f,map_location='cpu')
        langModel = lang_model.RNNModel(args.model, ntokens, args.emsize_lm, args.nhid, args.nlayers, args.dropout, args.dropouth, args.dropouti_lm, args.dropoute_lm, args.wdrop, args.tied_lm, pretrained_lm)
    del pretrained_lm



if args.cuda:
    model_gen = model_gen.cuda()
    criterion = criterion.cuda()
    criterion_reconst = criterion_reconst.cuda()
    if args.use_semantic_loss:
        criterion_sem = criterion_sem.cuda()
        modelSentEncoder = modelSentEncoder.cuda()
    if args.use_lm_loss:
        criterion_lm = criterion_lm.cuda()
        langModel = langModel.cuda()
		
###
params = list(model_gen.parameters()) + list(criterion.parameters()) + list(criterion_reconst.parameters())
params_gen = model_gen.parameters()
params_msg = list(model_gen.msg_out_mlp.parameters())

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

### conventions for real and fake labels during training ###
real_label = 1.0
fake_label = 0.0

###############################################################################
# Training code
###############################################################################

def evaluate(data_source, batch_size=10):
    # Turn on evaluation mode which disables dropout.
    model_gen.eval()
    if args.use_semantic_loss: 
        modelSentEncoder.eval()
    if args.use_lm_loss:
        langModel.eval()
		
    total_loss_msg = 0
	
    ntokens = len(corpus.dictionary)
    batches_count = 0
    for i in range(0, data_source.size(0) - args.bptt, args.bptt):
        data, msgs, targets = get_batch_different(data_source, i, args,all_msgs, evaluation=True)
        #get a batch of fake (edited) sequence from the generator
        fake_data_emb, fake_one_hot, fake_data_prob = model_gen.forward_sent(data,msgs,args.gumbel_temp)
        msg_out = model_gen.forward_msg_decode(fake_data_emb)
        data_emb = model_gen.forward_sent(data,msgs,args.gumbel_temp,only_embedding=True)

        #msg loss of the generator
        msg_loss = criterion(msg_out, msgs)

        total_loss_msg += msg_loss.data
        batches_count = batches_count + 1

    return total_loss_msg.item() / batches_count


def train():
    # Turn on training mode which enables dropout.
    freeze(model_gen)
    unfreeze(model_gen.msg_out_mlp)
    total_loss_msg = 0	
    start_time = time.time()
    ntokens = len(corpus.dictionary)
    batch, i = 0, 0
    while i < train_data.size(0) - 1 - 1:
        if not args.fixed_length:
            bptt = args.bptt if np.random.random() < 0.95 else args.bptt / 2.
            seq_len = max(5, int(np.random.normal(bptt, 5)))
            data, msgs, targets = get_batch_different(train_data, i, args,all_msgs, seq_len=seq_len)
        else:
            seq_len = args.bptt
            data, msgs, targets = get_batch_different(train_data, i, args, all_msgs, seq_len=None)

        model_gen.train()

        optimizer_gen.zero_grad()

        ####### Update lr #######
        if args.scheduler:
            optimizer_gen.param_groups[0]['lr'] = learing_rate_scheduler() 
		

	#get the embeddings from the generator network of the real 
        data_emb = model_gen.forward_sent(data,msgs,args.gumbel_temp,only_embedding=True)

        # Train with all-fake batch #
        # Generate batch of fake sequence #
        fake_data_emb, fake_one_hot, fake_data_prob = model_gen.forward_sent(data,msgs,args.gumbel_temp)
        dec_input = torch.matmul(fake_one_hot, model_gen.embeddings.weight).detach()
        msg_out = model_gen.forward_msg_decode(dec_input)

        ####### Update Generator Network #######
        # Maximize log(D(G(z)))
        # For the generator loss the labels are real #
        errG_msg = criterion(msg_out,msgs)

        errG = args.msg_weight*errG_msg

        errG.backward()
        # update the generator #
        optimizer_gen.step()

        # save losses #
        msg_losses.append(errG_msg.item())

        total_loss_msg += errG_msg.item()

        if batch % args.log_interval == 0 and batch > 0:
            cur_loss_msg = total_loss_msg / args.log_interval

            elapsed = time.time() - start_time
            print('| epoch {:3d} | {:5d}/{:5d} batches | gen lr {:05.5f} | ms/batch {:5.2f} | '
                    'msg loss {:5.2f}'.format(
                epoch, batch, len(train_data) // args.bptt, optimizer_gen.param_groups[0]['lr'],
                elapsed * 1000 / args.log_interval, cur_loss_msg))
            total_loss_msg = 0
            start_time = time.time()
        ###
        batch += 1
        i += seq_len
        global step_num
        step_num += 1
# Loop over epochs.
lr = args.lr
best_val_loss = []
G_losses = []
D_losses = []
msg_losses = []
sem_losses = []
stored_loss = 100000000
stored_loss_msg = 100000000
stored_loss_text = 100000000

# At any point you can hit Ctrl + C to break out of training early.
try:
    optimizer_gen = None
    # Ensure the optimizer is optimizing params, which includes both the model's weights as well as the criterion's weight (i.e. Adaptive Softmax)
    if args.optimizer == 'sgd':
        optimizer_gen = torch.optim.SGD(params_msg, lr=args.lr, weight_decay=args.wdecay)
    if args.optimizer == 'adam':
        optimizer_gen = torch.optim.Adam(params_msg, lr=learing_rate_scheduler() if args.scheduler else args.lr, betas=(args.beta1,args.beta2), weight_decay=args.wdecay)

    for epoch in range(1, args.epochs+1):
        epoch_start_time = time.time()
        train()
        if 't0' in optimizer_gen.param_groups[0]:
            tmp = {}
            for prm in model_gen.parameters():
                tmp[prm] = prm.data.clone()
                prm.data = optimizer_gen.state[prm]['ax'].clone()


            val_loss_msg2= evaluate(val_data, eval_batch_size)
            val_loss_gen_tot2 = val_loss_msg2

            print('-' * 89)
            print('| end of epoch {:3d} | time: {:5.2f}s | '
                ' val msg loss {:5.2f} '.format(
                    epoch, (time.time() - epoch_start_time), val_loss_msg2))
            print('-' * 89)
            log_file_loss_val.write( str(val_loss_msg2) + '\n')
            log_file_loss_val.flush()


            if val_loss_gen_tot2 < stored_loss:
                model_save(args.save)
                print('Saving Averaged!')
                stored_loss = val_loss_gen_tot2

            for prm in model_gen.parameters():
                prm.data = tmp[prm].clone()


        else:
            val_loss_msg = evaluate(val_data, eval_batch_size)
            val_loss_gen_tot =  val_loss_msg 
            print('-' * 89)
            print('| end of epoch {:3d} | time: {:5.2f}s | '
                ' val msg loss {:5.2f} |'.format(
                    epoch, (time.time() - epoch_start_time),val_loss_msg))
            print('-' * 89)
            log_file_loss_val.write( str(val_loss_msg)  + '\n')
            log_file_loss_val.flush()

            if val_loss_gen_tot < stored_loss:
                model_save(args.save)
                print('Saving model (new best generator validation)')
                stored_loss = val_loss_gen_tot
                stored_loss_msg = val_loss_msg
            if epoch % args.save_interval == 0:
                model_save(args.save+'_interval')
                print('Saving model (intervals)')

            if args.optimizer == 'sgd' and 't0' not in optimizer_gen.param_groups[0] and (len(best_val_loss)>args.nonmono and val_loss_gen_tot > min(best_val_loss[:-args.nonmono])): #modified by jz
                print('Switching to ASGD')
                optimizer_gen = torch.optim.ASGD(params_msg, lr=args.lr, t0=0, lambd=0., weight_decay=args.wdecay)

            if args.optimizer == 'sgd' and epoch in args.when:
                print('Saving model before learning rate decreased')
                model_save('{}.e{}'.format(args.save, epoch))
                print('Dividing learning rate by 10')
                optimizer_gen.param_groups[0]['lr'] /= 10.

            best_val_loss.append(val_loss_gen_tot)

except KeyboardInterrupt:
    print('-' * 89)
    print('Exiting from training early')

# Load the best saved model.
model_load(args.save)
if args.cuda:
    model_gen = model_gen.cuda()
    criterion = criterion.cuda()
    criterion_reconst = criterion_reconst.cuda()
    if args.use_semantic_loss:
        criterion_sem = criterion_sem.cuda()
        modelSentEncoder = modelSentEncoder.cuda()


# Run on test data.
test_loss_msg = evaluate(test_data, test_batch_size)

print('-' * 89)
print('| End of training | test msg loss {:5.2f}'.format(test_loss_msg))
print('-' * 89)

