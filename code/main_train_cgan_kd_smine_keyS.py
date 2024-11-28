import argparse
import time
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import os 
import sys
import model_mt_autoenc_cce_cgan_keyS as model_mt_autoenc_cce_cgan
import model_denoise_autoenc_attack
import model_mine
import lang_model
import mine
import utils
import torch.autograd as autograd
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:512"

from utils import batchify, get_batch_different, generate_msgs, repackage_hidden, get_batch_different_cgan, calculate_label
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
parser.add_argument('--dont_share_encoder', dest='shared_encoder', default=True, action='store_false',
                    help='If the message encoder and language encoder will share weights')

#adv. transformer arguments
parser.add_argument('--adv_attn_heads', type=int, default=4,
                    help='The number of attention heads in the adversary transformer')
parser.add_argument('--adv_encoding_layers', type=int, default=3,
                    help='The number of encoding layers in the adversary transformer')

#gumbel softmax arguments
parser.add_argument('--gumbel_temp', type=float, default=0.5,
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

parser.add_argument('--D1_thres', type=float, default=0.0,
                    help='threshold to punish errG_reconst')
parser.add_argument('--gan_D1_thres', type=float, default=1.0,
                    help='threshold to punish errG_reconst')

#GAN arguments			
parser.add_argument('--discr_interval', type=int, default=1,
                    help='when to update the discriminator')
parser.add_argument('--autoenc_path', type=str, default='',
                    help='path of the autoencoder path to use as init to the generator, in case the model is pretrained as autoencoder only')
parser.add_argument('--gen_weight', type=float, default=2,
                    help='The factor multiplied with the gen loss')
parser.add_argument('--msg_weight', type=float, default=1.0,
                    help='weight of errG_msg')
parser.add_argument('--msg_label_weight', type=float, default=1.0,
                    help='weight of errG_msg_label')
parser.add_argument('--reconst_weight', type=float, default=1.0,
                    help='weight of reconstruction loss in errG')
parser.add_argument('--xsmi_weight',type=float, default=1.0,
                    help='weight of errG_mi')

parser.add_argument('--atk_mi_weight', type=float, default=1.0,
                    help='MINE weight in attacker loss')
parser.add_argument('--atk_reconst_weight', type=float, default=1.0,
                    help='reconst weight in attacker loss')

parser.add_argument('--atk_path', type=str, default='',
                    help='path of attacker model')

parser.add_argument('--atk_msg_weight', type=float, default=1,
                    help="weight of errA_msg in total errA")
parser.add_argument('--pos_drop', type=float, default=0.1)
# BERT KD options
parser.add_argument('--bert_kd', action='store_true',
                    help='use BERT KD for training')
# KD hyper params
parser.add_argument("--kd_alpha", default=0.5, type=float,
                    help="ratio between label and teacher loss")
parser.add_argument("--kd_beta", default=0, type=float,
                    help="ratio between label and teacher loss")
# where is KD
parser.add_argument('--kd_data', type=str, default='',
                    help='path of kd data')
#use saved corpus
parser.add_argument('--saved_corpus', type=str, default='',
                    help='path of corpus to resume')
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
        torch.save([model_gen, criterion, criterion_reconst, criterion_label, optimizer_gen], f)
    with open(fn+'_disc.pt', 'wb') as f:
        torch.save([model_disc, criterion, criterion_reconst, criterion_label, optimizer_disc], f)
    with open(fn+'_atk.pt', 'wb') as f:
        torch.save([model_atk,optimizer_atk], f)
    with open(fn+'_mine_xys.pt', 'wb') as f:
        torch.save(mine_xys,f)
    with open(fn+'_mine_xs.pt', 'wb') as f:
        torch.save(mine_xs,f)

def model_load(fn):
    global model_gen, model_disc, criterion, criterion_reconst, criterion_label, optimizer_gen, optimizer_disc
    with open(fn+'_gen.pt', 'rb') as f:
        model_gen, criterion, criterion_reconst, criterion_label, optimizer_gen = torch.load(f,map_location='cpu')
    with open(fn+'_disc.pt', 'rb') as f:
        model_disc, criterion, criterion_reconst, criterion_label, optimizer_disc = torch.load(f,map_location='cpu')
		
import os
import hashlib
from data_bert import BertTokenizedCorpus,Dictionary
print('Producing dataset...')
if args.saved_corpus:
    print('Using saved corpus')
    corpus = torch.load(args.saved_corpus)
else:
    print('Generating new corpus')
    fn = 'corpus.{}.data'.format(hashlib.md5(args.data.encode()).hexdigest())
    corpus = BertTokenizedCorpus(args.data)
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
criterion_label = None
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
discr_step_num = 1

def learing_rate_scheduler():
    d_model = args.emsize
    warm_up = args.warm_up
    lr = np.power(d_model, -0.8) * min(np.power(step_num, -0.5), step_num*np.power(warm_up, -1.5))
    return lr

def learing_rate_disc_scheduler():
    d_model = args.emsize
    warm_up = args.warm_up
    lr = np.power(d_model, -1.1) * min(np.power(discr_step_num, -0.5), discr_step_num*np.power(warm_up, -1.5))
    return lr
	
if args.resume:
    all_msgs = np.loadtxt('msgs.txt')
    print('Resuming model ...')
    model_load(args.resume) 
    optimizer_gen.param_groups[0]['lr'] = learing_rate_scheduler() if args.scheduler else args.lr
    # optimizer_disc.param_groups[0]['lr'] = learing_rate_disc_scheduler() if args.scheduler else args.lr
else:
    ### generate random msgs ###
    all_msgs = generate_msgs(args)
    model_gen = model_mt_autoenc_cce_cgan.TranslatorGeneratorModel(ntokens, args.emsize, args.msg_len, args.msg_in_mlp_layers , args.msg_in_mlp_nodes, args.encoding_layers, args.dropout_transformer, args.dropouti, args.dropoute, args.tied, args.shared_encoder, args.attn_heads,autoenc_model)
    model_disc = model_mt_autoenc_cce_cgan.TranslatorDiscriminatorModel(args.emsize, args.adv_encoding_layers, args.dropout_transformer, args.adv_attn_heads, args.dropouti, args.msg_len)
    for p in model_disc.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)

# model_atk = model_denoise_autoenc_attack.AutoencModel(ntokens, args.emsize, args.encoding_layers, args.pos_drop, args.dropout_transformer, args.dropouti, args.dropoute, args.tied, args.attn_heads).cuda()
atk_path = args.atk_path if args.atk_path != '' else args.resume+'_atk.pt'
model_atk = torch.load(atk_path)[0].cuda()
mine_xys = mine.Mine_withKey(model_mine.MI_3feat(args.emsize, args.dropout_transformer,args.attn_heads, args.encoding_layers), loss='mine').cuda()
mine_xs  = mine.Mine(model_mine.MI_2feat(args.emsize, args.dropout_transformer,args.attn_heads, args.encoding_layers), loss='mine').cuda()
###
if not criterion:
    criterion =  nn.BCEWithLogitsLoss()
if not criterion_label:
    criterion_label = nn.BCEWithLogitsLoss()
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

## bert_kd model ##
if args.bert_kd:
    train_kd = torch.load(args.kd_data+'train.data',map_location='cpu')
    train_kd_prob = train_kd['prob'].detach()
    train_kd_idx = train_kd['idx'].long().detach()
    valid_kd = torch.load(args.kd_data+'valid.data',map_location='cpu')
    valid_kd_prob = valid_kd['prob'].detach()
    valid_kd_idx = valid_kd['idx'].long().detach()
    test_kd = torch.load(args.kd_data+'test.data',map_location='cpu')
    test_kd_prob = test_kd['prob'].detach()
    test_kd_idx = test_kd['idx'].long().detach()
else:
    train_kd_prob = None
    train_kd_idx = None
    valid_kd_prob = None
    valid_kd_idx = None
    test_kd_prob = None
    test_kd_idx = None

if args.cuda:
    model_gen = model_gen.cuda()
    model_disc = model_disc.cuda()
    criterion = criterion.cuda()
    criterion_reconst = criterion_reconst.cuda()
    criterion_label = criterion_label.cuda()
    if args.use_semantic_loss:
        criterion_sem = criterion_sem.cuda()
        modelSentEncoder = modelSentEncoder.cuda()
    if args.use_lm_loss:
        criterion_lm = criterion_lm.cuda()
        langModel = langModel.cuda()
    if args.bert_kd:
        train_kd_prob = train_kd_prob.cuda()
        train_kd_idx = train_kd_idx.cuda()
        valid_kd_prob = valid_kd_prob.cuda()
        valid_kd_idx = valid_kd_idx.cuda()
        test_kd_prob = test_kd_prob.cuda()
        test_kd_idx = test_kd_idx.cuda()

###
params = list(model_gen.parameters()) + list(criterion.parameters()) + list(criterion_reconst.parameters()) + list(criterion_label.parameters()) + list(model_atk.parameters()) + list(mine_xys.parameters())+list(mine_xs.parameters()) + list(model_disc.parameters())
params_gen = model_gen.parameters()
params_disc = model_disc.parameters()

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

def evaluate(data_source, batch_size=10, kd_prob_set=None, kd_idx_set=None):
    # Turn on evaluation mode which disables dropout.
    model_gen.eval()
    model_disc.eval()
    if args.use_semantic_loss: 
        modelSentEncoder.eval()
    if args.use_lm_loss:
        langModel.eval()
    total_loss_gen = 0
    total_loss_disc = 0

    total_loss_reconst = 0
    total_loss_msg = 0
    total_loss_msg_label = 0
    total_mine_xs = 0
    total_mine_xys = 0
    total_loss_lm = 0
    total_loss_sem = 0
    total_loss_msg_1 = 0
    total_loss_msg_2 = 0
    total_loss_kd = 0
	
    ntokens = len(corpus.dictionary)
    batches_count = 0
    if args.use_lm_loss:
        hidden = langModel.init_hidden(batch_size)
    with torch.no_grad():
        for i in range(0, data_source.size(0) - args.bptt, args.bptt):
            data, msgs, targets, labels_gt = get_batch_different_cgan(data_source, i, args,all_msgs, evaluation=True)
            data_emb = model_gen.forward_sent(data,msgs,args.gumbel_temp,only_embedding=True)


            # Generate batch of fake sequence #
            # fake_data_emb, fake_one_hot, fake_data_prob = model_gen.forward_sent(data,msgs,args.gumbel_temp)
            fake_sent_msg_embedding, fake_encoder_out = model_gen.forward_sent_encoder(data,msgs, args.gumbel_temp, only_embedding=False)
            fake_data_emb, fake_one_hot, fake_data_prob = model_gen.forward_sent_decoder(fake_sent_msg_embedding, data, fake_encoder_out, args.gumbel_temp)
            
            keys = data_emb.detach()

            # msg_gt_labels = list(map(calculate_label, msgs))
            # labels_gt = torch.Tensor(msg_gt_labels).to(torch.int64)         ### ground-truth labels
            # labels_gt = F.one_hot(labels_gt, num_classes = 2 ** args.msg_len).float()
            # if args.cuda:
            #     labels_gt = labels_gt.cuda()

            real_out = model_disc(data_emb, labels_gt)
            label = torch.full( (data.size(1),1), real_label)
            if args.cuda:
                label = label.cuda()

            errD_real = real_out.sum()
            #get prediction (and the loss) of the discriminator on the fake sequence.
            fake_out = model_disc(fake_data_emb.detach(), labels_gt)
            label.fill_(fake_label)
            errD_fake = fake_out.sum()
            errD = errD_fake - errD_real
            # errD = errD_real + errD_fake
            
            #generator loss
            label.fill_(real_label) 
            errG_disc = torch.relu(-errD - args.gan_D1_thres)
            # errG_disc = criterion(fake_out,label) if errD_fake.data+errD_real.data < args.gan_D1_thres else torch.tensor(0.0,device='cuda')

            atk_data_prob, atk_data_emb = model_atk.forward_sent(fake_data_emb, fake_data_emb, args.gumbel_temp, True, True)

            #reconstruction loss 
            reconst_loss = criterion_reconst(fake_data_prob,data.view(-1))
            total_loss_reconst += reconst_loss.data

            atk_data_hot = F.gumbel_softmax(F.log_softmax(atk_data_prob,dim=-1), tau = args.gumbel_temp, hard=True)
            msg_dec_input = torch.mm(atk_data_hot,model_gen.embeddings.weight).view(*fake_data_emb.shape)
            msg_out, labels_out = model_gen.forward_msg_decode(msg_dec_input, keys)
            msg_out_2, labels_out_2 = model_gen.forward_msg_decode(fake_data_emb, keys)
            msg_loss_1 = criterion(msg_out, msgs)
            msg_loss_2 = criterion(msg_out_2, msgs)
            msg_loss = msg_loss_1 + msg_loss_2

            #### msg_label_loss here ####
            msg_label_loss = criterion_label(labels_out, labels_gt)
            total_loss_msg_label += msg_label_loss
            ### TODO: msg_label_loss = ...
            ###       total_loss_msg_label += msg_label_loss

            total_loss_msg += msg_loss
            total_loss_msg_1 += msg_loss_1
            total_loss_msg_2 += msg_loss_2
            total_loss_gen +=  errG_disc
            total_loss_disc +=  errD

            fake_data_feat = model_gen.pos_encoder(fake_data_emb)
            fake_data_feat = model_gen.sent_encoder(fake_data_feat).mean(dim=0)
            data_feat = model_gen.pos_encoder(data_emb)
            data_feat = model_gen.sent_encoder(data_feat).mean(dim=0)
            atk_data_feat = model_gen.pos_encoder(atk_data_emb)
            atk_data_feat = model_gen.sent_encoder(atk_data_feat).mean(dim=0)
            # key_data_emb = atk_data_emb.clone()
            # key_data_emb[keys.mask] = keys.emb[keys.mask]
            # key_data_feat = model_gen.pos_encoder(key_data_emb)
            # key_data_feat = model_gen.sent_encoder(key_data_feat).mean(dim=0)
            key_data_feat = data_feat.clone()




            total_mine_xs += -mine_xs(fake_data_feat, data_feat)
            total_mine_xys += - mine_xys(fake_data_feat,atk_data_feat, data_feat)

            #semantic loss
            if args.use_semantic_loss: 
                orig_sem_emb = modelSentEncoder.forward_encode_nopad(data)
                fake_sem_emb = modelSentEncoder.forward_encode_nopad(fake_one_hot,one_hot=True)
                sem_loss = criterion_sem(orig_sem_emb,fake_sem_emb)
                total_loss_sem += sem_loss.data

            #lm loss of the generator
            if args.use_lm_loss:
                lm_targets = fake_one_hot[1:fake_one_hot.size(0)]
                lm_targets = torch.argmax(lm_targets,dim=-1)
                lm_targets = lm_targets.view(lm_targets.size(0)*lm_targets.size(1),)
                lm_inputs = fake_one_hot[0:fake_one_hot.size(0)-1]
                lm_out,hidden = langModel(lm_inputs,hidden, decode=True,one_hot=True)
                lm_loss = criterion_lm(lm_out,lm_targets)
                total_loss_lm += lm_loss.data
                hidden = repackage_hidden(hidden)
                
            if args.bert_kd:
                #Compute teacher's distribution
                seq_len_true = min(args.bptt, len(data_source) - 1 - i)
                loss_kd = -(kd_prob_set[i:i+seq_len_true]*fake_data_prob.view(seq_len_true,batch_size,-1).gather(dim=-1,index=kd_idx_set[i:i+seq_len_true])).sum() / batch_size / seq_len_true

                total_loss_kd += loss_kd.item()

            batches_count = batches_count + 1	

    if args.use_semantic_loss: 
        total_loss_sem = total_loss_sem.item()
    if args.use_lm_loss: 
        total_loss_lm = total_loss_lm.item() 

    return total_loss_reconst.item()/batches_count, total_loss_msg.item()/batches_count, total_loss_msg_label.item()/batches_count, total_mine_xs.item()/batches_count,total_mine_xys.item()/batches_count, total_loss_sem / batches_count, total_loss_lm/batches_count, total_loss_msg_1.item()/batches_count, total_loss_msg_2.item()/batches_count, total_loss_gen.item()/batches_count, total_loss_disc.item()/batches_count, total_loss_kd/batches_count

def calculate_gradient_penalty(model_disc, real_images, fake_images, labels_gt):
    eta = torch.FloatTensor(1,real_images.shape[1],1).uniform_(0,1)
    eta = eta.expand(*real_images.shape)
    eta = eta.cuda()

    interpolated = eta * real_images + ((1 - eta) * fake_images)

    # calculate probability of interpolated examples
    prob_interpolated = model_disc(interpolated, labels_gt)

    # calculate gradients of probabilities with respect to examples
    gradients = autograd.grad(outputs=prob_interpolated, inputs=interpolated,
                            grad_outputs=torch.ones(
                                prob_interpolated.size()).cuda(),
                            create_graph=True, retain_graph=True)[0]

    grad_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
    return grad_penalty

def train():
    # Turn on training mode which enables dropout.
    total_loss_gen = 0
    total_loss_msg = 0
    total_loss_msg_label = 0
    total_loss_sem = 0
    total_loss_disc = 0
    total_loss_reconst = 0
    total_loss_lm = 0
    total_loss_sxmi = 0
    total_loss_msg_1 = 0
    total_loss_msg_2 = 0
    total_loss_kd = 0
    if args.use_lm_loss:
        hidden = langModel.init_hidden(args.batch_size)
		
    start_time = time.time()
    ntokens = len(corpus.dictionary)
    batch, i = 0, 0
    while i < train_data.size(0) - 1 - 1:
        if not args.fixed_length:
            bptt = args.bptt if np.random.random() < 0.95 else args.bptt / 2.
            seq_len = max(5, int(np.random.normal(bptt, 5)))
            data, msgs, targets, labels_gt = get_batch_different_cgan(train_data, i, args,all_msgs, seq_len=seq_len)
        else:
            seq_len = args.bptt
            data, msgs, targets, labels_gt = get_batch_different_cgan(train_data, i, args, all_msgs, seq_len=None)

        # msg_gt_labels = list(map(calculate_label, msgs))
        # labels_gt = torch.Tensor(msg_gt_labels).to(torch.int64)         ### ground-truth labels
        # labels_gt = F.one_hot(labels_gt, num_classes = 2 ** args.msg_len).float()

        utils.unfreeze(model_gen) # 保证建立计算图时，model_gen require grad
        model_gen.train()
        model_disc.train()

        if args.use_semantic_loss: 
            modelSentEncoder.train()
            #set parameters trainable to false.
            for p in modelSentEncoder.parameters(): #reset requires_grad
                p.requires_grad = False #they are set to False below in the generator update

        if args.use_lm_loss:
            langModel.train()
            #set parameters trainable to false.
            for p in langModel.parameters(): #reset requires_grad
                p.requires_grad = False #they are set to False below in the generator update
            hidden = repackage_hidden(hidden)

        optimizer_gen.zero_grad()
        optimizer_disc.zero_grad()

        ####### Update lr #######
        if args.scheduler:
            optimizer_gen.param_groups[0]['lr'] = learing_rate_scheduler() 
            # optimizer_disc.param_groups[0]['lr'] = learing_rate_disc_scheduler() 

	#get the embeddings from the generator network of the real 
        data_emb = model_gen.forward_sent(data,msgs,args.gumbel_temp,only_embedding=True)
        ####### Update Disc Network ####### 
        # Maximize log (D(x) + log (1 - D(G(z))) #
        # Train with all-real batch #
        
        label = torch.full( (data.size(1),1), real_label)
        if args.cuda:
            label = label.cuda()
            # labels_gt = labels_gt.cuda()
        real_out = model_disc(data_emb, labels_gt)

        errD_real = real_out.sum()

        fake_sent_msg_embedding, fake_encoder_out = model_gen.forward_sent_encoder(data,msgs, args.gumbel_temp, only_embedding=False)
        fake_data_emb, fake_one_hot, fake_data_prob = model_gen.forward_sent_decoder(fake_sent_msg_embedding, data, fake_encoder_out, args.gumbel_temp)

        fake_out = model_disc(fake_data_emb.detach(), labels_gt)
        label.fill_(fake_label)
        errD_fake = fake_out.sum()
        lam = 10.0
        GP = calculate_gradient_penalty(model_disc,data_emb,fake_data_emb.detach(), labels_gt)
        # add the gradients #
        errD = errD_fake - errD_real +  GP * lam
        # print("errD_fake", errD_fake)
        # print("errD_real", errD_real)
        # print("GP", GP)
        # update the discriminator #
        if batch % args.discr_interval == 0 and batch > 0:
            optimizer_disc.step()

        # Generate batch of fake sequence #
        # fake_data_emb, fake_one_hot, fake_data_prob = model_gen.forward_sent(data,msgs,args.gumbel_temp)
        fake_sent_msg_embedding, fake_encoder_out = model_gen.forward_sent_encoder(data,msgs, args.gumbel_temp, only_embedding=False)
        fake_data_emb, fake_one_hot, fake_data_prob = model_gen.forward_sent_decoder(fake_sent_msg_embedding, data, fake_encoder_out, args.gumbel_temp)
        # mask = data!=torch.argmax(fake_one_hot, dim=-1)
        keys = data_emb.detach()
        # import pdb;pdb.set_trace()
        # NOTE: fake_data_emb = torch.mm(sent_decoded_vocab_hot,self.embeddings.weight), after sampling!
        atk_input = torch.matmul(fake_one_hot, model_atk.embeddings.weight)
        atk_data_prob, atk_data_emb = model_atk.forward_sent(atk_input, atk_input, args.gumbel_temp, True, True)
        # atk_input = torch.matmul(fake_data_prob.view(*fake_one_hot.shape), model_atk.embeddings.weight) # used for sampling
        # atk_data_prob = model_atk.forward_sent(atk_input, atk_input, args.gumbel_temp, is_embedding=True, return_feature=False)
        # atk_data_emb = torch.matmul(atk_data_prob.view(*fake_one_hot.shape), model_gen.embeddings.weight)

        # atk_data_hot = F.gumbel_softmax(F.log_softmax(atk_data_prob,dim=-1), tau = args.gumbel_temp, hard=True)
        # atk_data_emb = torch.mm(atk_data_hot,model_gen.embeddings.weight).view(*fake_data_emb.shape)

        ###### updata MINE ######
        utils.unfreeze(mine_xys)
        utils.unfreeze(mine_xs)
        
        with torch.no_grad():
            fake_data_feat = model_gen.pos_encoder(fake_data_emb)
            fake_data_feat = model_gen.sent_encoder(fake_data_feat).mean(dim=0)
            data_feat = model_gen.pos_encoder(data_emb)
            data_feat = model_gen.sent_encoder(data_feat).mean(dim=0)
            atk_data_feat = model_gen.pos_encoder(atk_data_emb)
            atk_data_feat = model_gen.sent_encoder(atk_data_feat).mean(dim=0)
            # key_data_emb = atk_data_emb.detach().clone()
            # key_data_emb[keys.mask] = keys.emb[keys.mask]
            # key_data_feat = model_gen.pos_encoder(key_data_emb)
            # key_data_feat = model_gen.sent_encoder(key_data_feat).mean(dim=0)
            key_data_feat = data_feat.clone()

        mine_xys.optimize(fake_data_feat.detach(), atk_data_feat.detach(), data_feat.detach(), iters=2, batch_size=fake_data_emb.size(1),opt=optimizer_mine_xys) ### : batch first! 
        mine_xs.optimize(fake_data_feat.detach(), data_feat.detach(), iters=2, batch_size=fake_data_emb.size(1),opt=optimizer_mine_xs) ### : batch first! 
        utils.freeze(mine_xys)
        utils.freeze(mine_xs)
        
        atk_data_hot = F.gumbel_softmax(F.log_softmax(atk_data_prob,dim=-1), tau = args.gumbel_temp, hard=True)
        msg_dec_input = torch.mm(atk_data_hot,model_gen.embeddings.weight).view(*fake_data_emb.shape)
        msg_out, labels_out = model_gen.forward_msg_decode(msg_dec_input, keys)

        ####### Update Generator Network #######
        optimizer_gen.zero_grad()
        errG_reconst = criterion_reconst(fake_data_prob,data.view(-1)) #S, X distortion
        total_loss_reconst += errG_reconst.data
        # errG_mi: I(U;Y) - I(U;S) replaced by I(X;Y|S) = I(X;Y,S) - I(X;S) (maximize)
        # errG_mi = -(mine_xs(fake_sent_msg_embedding, data_emb.permute((1,0,2))) - mine_xys(fake_sent_msg_embedding,atk_data_emb.permute((1,0,2)))) # min -(I(U;Y) - I(U;S)) # J =  I(U;Y) - I(U;S)

        with torch.no_grad():
            fake_data_feat = model_gen.pos_encoder(fake_data_emb)
            fake_data_feat = model_gen.sent_encoder(fake_data_feat).mean(dim=0)
            data_feat = model_gen.pos_encoder(data_emb)
            data_feat = model_gen.sent_encoder(data_feat).mean(dim=0)
            atk_data_feat = model_gen.pos_encoder(atk_data_emb)
            atk_data_feat = model_gen.sent_encoder(atk_data_feat).mean(dim=0)
            # key_data_emb = atk_data_emb.clone()
            # key_data_emb[keys.mask] = keys.emb[keys.mask]
            # key_data_feat = model_gen.pos_encoder(key_data_emb)
            # key_data_feat = model_gen.sent_encoder(key_data_feat).mean(dim=0)
            key_data_feat = data_feat.clone()

        errG_mi = -(-mine_xys(fake_data_feat,atk_data_feat, data_feat) + mine_xs(fake_data_feat, data_feat)) 

        data_emb = model_gen.forward_sent(data,msgs,args.gumbel_temp,only_embedding=True)
        msg_out_clean, labels_out_clean = model_gen.forward_msg_decode(fake_data_emb, keys)
        errG_msg_1 = criterion(msg_out, msgs) 
        errG_msg_2 = criterion(msg_out_clean, msgs)
        errG_msg = errG_msg_1 + errG_msg_2


        ### TODO: errG_msg_label = ...
        # print(msg_out)
        # msg_out_labels = list(map(calculate_label, msg_out))
        # labels_out = torch.Tensor(msg_out_labels).to(torch.int64)
        # print(labels_out)
        # labels_out = F.one_hot(labels_out, num_classes = 2 ** args.msg_len)
        errG_msg_label = criterion_label(labels_out, labels_gt)

        # import pdb;pdb.set_trace()
        if args.use_semantic_loss: 
            # Compute sentence embedding #
            orig_sent_emb = modelSentEncoder.forward_encode_nopad(data)
            fake_sent_emb = modelSentEncoder.forward_encode_nopad(fake_one_hot, one_hot=True)
            errG_sem = criterion_sem(orig_sent_emb,fake_sent_emb) 
            errG = args.reconst_weight*errG_reconst + args.xsmi_weight * errG_mi + args.msg_weight*errG_msg + args.msg_label_weight*errG_msg_label + args.sem_weight*errG_sem # + args.gen_weight*errG_disc
            sem_losses.append(errG_sem.item())
            total_loss_sem += errG_sem.item()
        else:
            errG = args.reconst_weight*errG_reconst + args.xsmi_weight * errG_mi + args.msg_weight*errG_msg + args.msg_label_weight*errG_msg_label # + args.gen_weight*errG_disc

        # Maximize log(D(G(z)))
        # For the generator loss the labels are real #
        label.fill_(real_label) 
        # Classify with the updated discriminator #
        fake_out2 = model_disc(fake_data_emb, labels_gt)
        errG_disc = torch.relu(errD_real.detach()-fake_out2.sum() - args.gan_D1_thres)
        errG = errG + args.gen_weight * errG_disc

        del labels_gt

        if args.use_lm_loss:
            lm_targets = fake_one_hot[1:fake_one_hot.size(0)]
            lm_targets = torch.argmax(lm_targets,dim=-1)
            lm_targets = lm_targets.view(lm_targets.size(0)*lm_targets.size(1),)
            lm_inputs = fake_one_hot[0:fake_one_hot.size(0)-1]
            lm_out,hidden = langModel(lm_inputs,hidden, decode=True,one_hot=True)
            lm_loss = criterion_lm(lm_out,lm_targets)
            errG = errG + args.lm_weight*lm_loss
            total_loss_lm += lm_loss.item()	
            hidden = repackage_hidden(hidden)
            
        if args.bert_kd:
            #Compute teacher's distribution
            kd_alpha = args.kd_alpha
            kd_beta = args.kd_beta
            seq_len_true = min(seq_len if seq_len else args.bptt, len(train_data) - 1 - i)
            loss_kd = -(train_kd_prob[i:i+seq_len_true]*fake_data_prob.view(seq_len_true,args.batch_size,-1).gather(dim=-1,index=train_kd_idx[i:i+seq_len_true])).sum() / args.batch_size / seq_len_true
            # print("kdloss:",loss_kd.item())
            errG = errG* kd_beta + loss_kd * kd_alpha
            total_loss_kd += loss_kd.item()


        errG.backward()
        # update the generator #
        optimizer_gen.step()
        utils.freeze(model_gen)

        total_loss_reconst += errG_reconst.data
        total_loss_sxmi += errG_mi.data
        total_loss_msg += errG_msg.data
        total_loss_msg_label += errG_msg_label.data
        total_loss_msg_1 += errG_msg_1.data
        total_loss_msg_2 += errG_msg_2.data
        total_loss_disc += errD_real.data+errD_fake.data
        total_loss_gen += errG_disc.data


        ###### Update attacker ######
        utils.unfreeze(model_atk)
        optimizer_atk.zero_grad()
        # fake_data_emb, fake_one_hot, fake_data_prob = model_gen.forward_sent(data,msgs,args.gumbel_temp)
        fake_sent_msg_embedding, fake_encoder_out = model_gen.forward_sent_encoder(data,msgs, args.gumbel_temp, only_embedding=False)
        fake_data_emb, fake_one_hot, fake_data_prob = model_gen.forward_sent_decoder(fake_sent_msg_embedding, data, fake_encoder_out, args.gumbel_temp)

        # atk_data_prob, atk_data_emb = model_atk.forward_sent(fake_data_emb, fake_data_emb, args.gumbel_temp, is_embedding=True, return_feature=True)
        # atk_data_hot = F.gumbel_softmax(F.log_softmax(atk_data_prob,dim=-1), tau = args.gumbel_temp, hard=True)
        # atk_data_emb = torch.mm(atk_data_hot, model_gen.embeddings.weight).view(*fake_data_emb.shape)
        atk_input = torch.matmul(fake_one_hot, model_atk.embeddings.weight)
        atk_data_prob, atk_data_emb = model_atk.forward_sent(atk_input, atk_input, args.gumbel_temp, True, True)

        errA_reconst = criterion_reconst(atk_data_prob.view(-1, atk_data_prob.shape[-1]),data.view(-1))
        errA = args.atk_reconst_weight * errA_reconst
        errA.backward()
        optimizer_atk.step()
        utils.freeze(model_atk)

        if batch % args.log_interval == 0 and batch > 0:
            cur_loss_gen = total_loss_gen / args.log_interval
            cur_loss_disc = total_loss_disc / args.log_interval
            cur_loss_msg = total_loss_msg / args.log_interval
            cur_loss_msg_label = total_loss_msg_label / args.log_interval
            cur_loss_sem = total_loss_sem / args.log_interval
            cur_loss_reconst = total_loss_reconst / args.log_interval
            cur_loss_lm = total_loss_lm / args.log_interval
            cur_loss_msg_1 = total_loss_msg_1 / args.log_interval
            cur_loss_msg_2 = total_loss_msg_2 / args.log_interval
            cur_loss_kd = total_loss_kd / args.log_interval

            elapsed = time.time() - start_time
            print('| epoch {:3d} | {:5d}/{:5d} batches | gen lr {:05.5f} | ms/batch {:5.6f} | '
                    'msg loss {:5.6f} | msg label loss {:5.6f} | sem loss {:5.6f} | reconst loss {:5.6f} | lm loss {:5.6f} | errA_mi -- | errA_reconst {:5.6f}| msg_1 loss {:5.6f} | msg_2 loss {:5.6f}  | gen loss {:5.6f} | disc loss {:5.6f} | kd loss {:5.6f}'.format(
                epoch, batch, len(train_data) // args.bptt, optimizer_gen.param_groups[0]['lr'],
                elapsed * 1000 / args.log_interval, cur_loss_msg, cur_loss_msg_label, cur_loss_sem, cur_loss_reconst, cur_loss_lm, errA_reconst, cur_loss_msg_1, cur_loss_msg_2, cur_loss_gen, cur_loss_disc, cur_loss_kd))
            total_loss_gen = 0
            total_loss_msg = 0
            total_loss_msg_label = 0
            total_loss_disc = 0
            total_loss_sem = 0
            total_loss_reconst = 0
            total_loss_sxmi = 0
            total_loss_sxmi = 0
            total_loss_lm = 0
            total_loss_msg_1 = 0
            total_loss_msg_2 = 0
            total_loss_kd = 0
            start_time = time.time()
        ###
        batch += 1
        i += seq_len
        global step_num, discr_step_num
        step_num += 1
        discr_step_num += 1

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
    optimizer_disc = None
    # Ensure the optimizer is optimizing params, which includes both the model's weights as well as the criterion's weight (i.e. Adaptive Softmax)
    if args.optimizer == 'sgd':
        optimizer_gen = torch.optim.SGD(params_gen, lr=args.lr, weight_decay=args.wdecay)
        optimizer_disc = torch.optim.SGD(params_disc, lr=args.lr, weight_decay=args.wdecay)
        optimizer_atk = torch.optim.SGD(model_atk.parameters(), lr=args.lr, weight_decay=args.wdecay)
    if args.optimizer == 'adam':
        optimizer_gen = torch.optim.Adam(params_gen, lr=learing_rate_scheduler() if args.scheduler else args.lr, betas=(args.beta1,args.beta2), weight_decay=args.wdecay)
        optimizer_disc = torch.optim.Adam(params_disc, lr=learing_rate_disc_scheduler() if args.scheduler else args.disc_lr, betas=(args.beta1,args.beta2), weight_decay=args.wdecay)
        optimizer_atk = torch.optim.Adam(model_atk.parameters(), lr=learing_rate_scheduler() if args.scheduler else args.atk_lr, betas=(args.beta1,args.beta2), weight_decay=args.wdecay)
    optimizer_mine_xys = torch.optim.Adam(mine_xys.parameters(), lr=1e-4)
    optimizer_mine_xs = torch.optim.Adam(mine_xs.parameters(), lr=1e-4)
    for epoch in range(1, args.epochs+1):
        epoch_start_time = time.time()
        train()
        if 't0' in optimizer_gen.param_groups[0]:
            tmp = {}
            for prm in model_gen.parameters():
                tmp[prm] = prm.data.clone()
                prm.data = optimizer_gen.state[prm]['ax'].clone()

            tmp_disc = {}
            for prm in model_disc.parameters():
                tmp_disc[prm] = prm.data.clone()
                prm.data = optimizer_disc.state[prm]['ax'].clone()

            val_loss_reconst, val_loss_msg, val_loss_msg_label, val_loss_mine_us,val_loss_mine_uy, val_loss_sem, val_loss_lm, val_loss_msg_1, val_loss_msg_2, val_loss_gen, val_loss_disc, val_loss_kd= evaluate(val_data, eval_batch_size, valid_kd_prob, valid_kd_idx)
            val_loss_gen_tot2 = args.kd_beta*(args.reconst_weight * val_loss_reconst + args.msg_weight * val_loss_msg + args.xsmi_weight * (val_loss_mine_us - val_loss_mine_uy) + args.sem_weight * val_loss_sem + args.lm_weight * val_loss_lm + args.gen_weight * val_loss_gen) + args.kd_alpha*val_loss_kd

            print('-' * 89)
            print('| end of epoch {:3d} | time: {:5.6f}s | val_loss_reconst: {:5.6f} | val_loss_msg: {:5.6f} | val_loss_msg_label: {:5.6f} | val_loss_mine_us: {:5.6f} | val_loss_mine_uy: {:5.6f} | val_loss_sem: {:5.6f} | val_loss_lm: {:5.6f} | val_loss_msg_1: {:5.6f} | val_loss_msg_2: {:5.6f} | val_loss_gen: {:5.6f} | val_loss_kd: {:5.6f}'.format(
                    epoch, (time.time() - epoch_start_time),val_loss_reconst, val_loss_msg, val_loss_msg_label, val_loss_mine_us,val_loss_mine_uy, val_loss_sem, val_loss_lm, val_loss_msg_1, val_loss_msg_2, val_loss_gen, val_loss_kd))
            print('-' * 89)
            # log_file_loss_val.write(str(val_loss_gen2) + ', '+ str(val_loss_disc2) + ', '+ str(val_loss_msg2) + ', '+ str(val_loss_sem2) + ', '+ str(val_loss_reconst2) + ', '+ str(val_loss_lm2) + '\n')
            # log_file_loss_val.flush()


            if val_loss_gen_tot2 < stored_loss:
                model_save(args.save)
                print('Saving Averaged!')
                stored_loss = val_loss_gen_tot2

            for prm in model_gen.parameters():
                prm.data = tmp[prm].clone()

            for prm in model_disc.parameters():
                prm.data = tmp[prm].clone()

        else:
            val_loss_reconst, val_loss_msg, val_loss_msg_label, val_loss_mine_us,val_loss_mine_uy, val_loss_sem, val_loss_lm,val_loss_msg_1, val_loss_msg_2, val_loss_gen, val_loss_disc, val_loss_kd = evaluate(val_data, eval_batch_size, valid_kd_prob, valid_kd_idx)
            val_loss_gen_tot = args.kd_beta*(args.reconst_weight * val_loss_reconst + args.msg_weight * val_loss_msg + args.xsmi_weight * (val_loss_mine_us - val_loss_mine_uy)+ args.sem_weight * val_loss_sem + args.lm_weight * val_loss_lm + args.gen_weight * val_loss_gen) + args.kd_alpha*val_loss_kd
            val_loss_text = val_loss_reconst
            print('-' * 89)
            print('| end of epoch {:3d} | time: {:5.6f}s | val_loss_reconst: {:5.6f} | val_loss_msg: {:5.6f} | val_loss_msg_label: {:5.6f} | val_loss_mine_us: {:5.6f} | val_loss_mine_uy: {:5.6f} | val_loss_sem {:5.6f} |  val_loss_lm {:5.6f} | val_loss_msg_1: {:5.6f} | val_loss_msg_2: {:5.6f} | val_loss_gen: {:5.6f} | val_loss_disc: {:5.6f} | val kd loss {:5.2f}'.format(
                    epoch, (time.time() - epoch_start_time),val_loss_reconst, val_loss_msg, val_loss_msg_label, val_loss_mine_us,val_loss_mine_uy,val_loss_sem, val_loss_lm, val_loss_msg_1, val_loss_msg_2, val_loss_gen, val_loss_disc, val_loss_kd))
            print('-' * 89)

            if val_loss_gen_tot < stored_loss:
                model_save(args.save)
                print('Saving model (new best generator validation)')
                stored_loss = val_loss_gen_tot
            if val_loss_msg < stored_loss_msg:
                model_save(args.save+'_msg')
                print('Saving model (new best msg validation)')
                stored_loss_msg = val_loss_msg
            if val_loss_text < stored_loss_text:
                model_save(args.save+'_reconst')
                print('Saving model (new best reconstruct validation)')
                stored_loss_text = val_loss_text
            if epoch % args.save_interval == 0:
                model_save(args.save+'_interval')
                print('Saving model (intervals)')

            if args.optimizer == 'sgd' and 't0' not in optimizer_gen.param_groups[0] and (len(best_val_loss)>args.nonmono and val_loss_gen_tot > min(best_val_loss[:-args.nonmono])): #modified by jz
                print('Switching to ASGD')
                optimizer_gen = torch.optim.ASGD(model_gen.parameters(), lr=args.lr, t0=0, lambd=0., weight_decay=args.wdecay)
                optimizer_disc = torch.optim.ASGD(model_disc.parameters(), lr=args.lr, t0=0, lambd=0., weight_decay=args.wdecay)

            if args.optimizer == 'sgd' and epoch in args.when:
                print('Saving model before learning rate decreased')
                model_save('{}.e{}'.format(args.save, epoch))
                print('Dividing learning rate by 10')
                optimizer_gen.param_groups[0]['lr'] /= 10.
                optimizer_disc.param_groups[0]['lr'] /= 10.

            best_val_loss.append(val_loss_gen_tot)

except KeyboardInterrupt:
    print('-' * 89)
    print('Exiting from training early')

# Load the best saved model.
model_load(args.save)
if args.cuda:
    model_gen = model_gen.cuda()
    model_disc = model_disc.cuda()
    criterion = criterion.cuda()
    criterion_reconst = criterion_reconst.cuda()
    criterion_label = criterion_label.cuda()
    if args.use_semantic_loss:
        criterion_sem = criterion_sem.cuda()
        modelSentEncoder = modelSentEncoder.cuda()


# Run on test data.
test_loss_reconst, test_loss_msg, test_loss_msg_label, test_loss_mine_us,test_loss_mine_uy, test_loss_sem, test_loss_lm, test_loss_msg_1, test_loss_msg_2, test_loss_gen, test_loss_disc, test_loss_kd = evaluate(test_data, test_batch_size, test_kd_prob, test_kd_idx)

print('-' * 89)
print('| end of training  {:3d} | time: {:5.6f}s | test_loss_reconst: {:5.6f} | test_loss_msg: {:5.6f} | test_loss_msg_label: {:5.6f} | test_loss_mine_us: {:5.6f} | test_loss_mine_uy: {:5.6f} | test_loss_sem: {:5.6f} | test_loss_lm: {:5.6f} | test_loss_msg_1: {:5.6f} | test_loss_msg_2: {:5.6f}| test gen loss {:5.2f} | test disc loss {:5.2f} | test kd loss {:5.2f}'.format(
        epoch, (time.time() - epoch_start_time),test_loss_reconst, test_loss_msg, test_loss_msg_label, test_loss_mine_us,test_loss_mine_uy, test_loss_sem, test_loss_lm, test_loss_msg_1, test_loss_msg_2, test_loss_gen, test_loss_disc, test_loss_kd))
print('-' * 89)
