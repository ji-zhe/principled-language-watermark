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
from utils import batchify, repackage_hidden, get_batch_different, generate_msgs
from nltk.translate.meteor_score import meteor_score

parser = argparse.ArgumentParser(description='PyTorch PennTreeBank RNN/LSTM Language Model')
parser.add_argument('--data', type=str, default='data/wikitext-2/',
                    help='location of the data corpus')
					
parser.add_argument('--bptt', type=int, default=80,
                    help='sequence length')

parser.add_argument('--given_msg', type=list, default=[],
                    help='test against this msg only')
parser.add_argument('--seq_len', type=int, default=80)
parser.add_argument('--pct', type=float, default=0.05)
parser.add_argument('--attack_type', type=str, default='insertion')

parser.add_argument('--seed', type=int, default=1111,
                    help='random seed')

parser.add_argument('--cuda', action='store_false',
                    help='use CUDA')

randomhash = ''.join(str(time.time()).split('.'))

parser.add_argument('--gen_path', type=str,  default='0421_mine_2msgloss_atkAdaptMap_phase2_gen.pt',
                    help='path to the generator')
parser.add_argument('--disc_path', type=str,  default='WT2_mt_full_disc.pt',
                    help='path to the discriminator')
parser.add_argument('--autoenc_attack_path', type=str,  default='model1_denoise_autoenc_attack.pt',
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
parser.add_argument('--msg_len', type=int, default=4,
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

corpus = data.Corpus(args.data)

train_batch_size = 20
eval_batch_size = 1
test_batch_size = 1
train_data = batchify(corpus.train, train_batch_size, args)
val_data = batchify(corpus.valid, eval_batch_size, args)
test_data = batchify(corpus.test, test_batch_size, args)

ntokens = len(corpus.dictionary)

print('Args:', args)

criterion1 =  nn.NLLLoss()
criterion2 =  nn.BCEWithLogitsLoss()
if args.use_lm_loss:
    criterion_lm = nn.CrossEntropyLoss() 

if args.use_lm_loss:
    with open(args.lm_ckpt, 'rb') as f:
        pretrained_lm, _,_ = torch.load(f,map_location='cpu')
        langModel = lang_model.RNNModel(args.model, ntokens, args.emsize_lm, args.nhid, args.nlayers, args.dropout, args.dropouth, args.dropouti_lm, args.dropoute_lm, args.wdrop, args.tied_lm, pretrained_lm)
    del pretrained_lm
    if args.cuda:
        langModel = langModel.cuda()
        criterion_lm = criterion_lm.cuda()

### generate random msgs ###
all_msgs = generate_msgs(args)
print(all_msgs)

if not args.given_msg == []:
    all_msgs = [int(i) for i in args.given_msg]
    all_msgs = np.asarray(all_msgs)
    all_msgs = all_msgs.reshape([1,args.msg_len])



def random_cut_sequence(sequence, limit=10):

    rand_cut_start = np.random.randint(low=0, high=limit)
    rand_cut_end = sequence.size(0) - np.random.randint(low=0, high=limit)

    sequence = sequence[rand_cut_start:rand_cut_end, :]
    new_seq_len = rand_cut_end -  rand_cut_start 
    #print(sequence.size())
    return sequence

def compare_msg_whole(msgs,msg_out):
    correct = np.count_nonzero(np.sum(np.equal(msgs.detach().cpu().numpy().astype(int),msg_out.detach().cpu().numpy().astype(int)),axis=1)==args.msg_len*args.msgs_segment)
    return correct
	
def compare_msg_bits(msgs,msg_out):
    correct = np.count_nonzero(np.equal(msgs.detach().cpu().numpy().astype(int),msg_out.detach().cpu().numpy().astype(int))==True)
    return correct
	
#def get_idx_from_logits(sequence,seq_len,bsz):
#    sequence = sequence.view(seq_len,bsz,sequence.size(1))
#    sequence_idx = torch.argmax(sequence, dim=-1)
#    return sequence_idx

	
	
def get_idx_from_logits(sequence,seq_len,bsz):
    m = nn.Softmax(dim=-1)
    sequence = sequence.view(seq_len,bsz,sequence.size(1))
    sequence = m(sequence)    
    sequence_idx = torch.argmax(sequence, dim=-1)
    return sequence_idx

### conventions for real and fake labels during training ###
real_label = 1
fake_label = 0

sbert_model = SentenceTransformer('bert-base-nli-mean-tokens') 
#convert word ids to text	
def convert_idx_to_words(idx):
    batch_list = []
    for i in range(0,idx.size(1)):
        sent_list = []
        for j in range(0,idx.size(0)):
            sent_list.append(corpus.dictionary.idx2word[idx[j,i]])
        batch_list.append(sent_list)
    return batch_list

def convert_words_to_idx(words):
    idx = torch.zeros(len(words),dtype=torch.long)
    for i in range(0,len(words)):
        idx[i] = corpus.dictionary.word2idx.get(words[i],corpus.dictionary.word2idx['<unk>'])
    return idx

def noisy_sampling(sent_encoder_out,both_embeddings,data):
    candidates_emb = []
    candidates_one_hot = []
    candidates_soft_prob = []
    for i in range(0,args.samples_num):
        with torch.no_grad():
            sent_out_emb, sent_out_hot, sent_out_soft = model_gen.forward_sent_decoder(both_embeddings, data, sent_encoder_out, args.gumbel_temp)
            candidates_emb.append(sent_out_emb)
            candidates_one_hot.append(sent_out_hot)
            candidates_soft_prob.append(sent_out_soft)			
    return candidates_emb,candidates_one_hot,candidates_soft_prob


def evaluate(data_source, out_file, batch_size=1, on_train=False):
    # Turn on evaluation mode which disables dropout.
    model_gen.eval()
    model_disc.eval()
    model_autoenc_attack.eval()
	
    total_loss_lm = 0
    ntokens = len(corpus.dictionary)
    tot_count = 0
    correct_msg_count = 0
    tot_count_bits = 0
    correct_msg_count_bits = 0
    sig = nn.Sigmoid()
    batch_count = 0
    meteor_tot = 0
    l2_distances = 0
    bert_diff = [0 for i in range(0,args.samples_num)]
    long_msg_count = 0

    p_value = []
    real_correct = 0
    fake_correct = 0 
    y_out = []
    y_label = []
    all_msg = torch.tensor(np.loadtxt(f"test_0421_mine_2msgloss_atkAdaptMap_phase2_cap{args.seq_len}_msg.txt", dtype=int)).cuda()
    long_msg = all_msg.view(-1)
    long_msg_out = torch.tensor([],dtype=torch.float).cuda()
    with open(f'/home/jizhe/hqsq/nlp-watermarking/data/ins_against_OURS_seq{args.seq_len}_PCT{args.pct}/ours-novels-corrupted-{args.attack_type}.txt', 'r') as f:
    # with open(f'test_0421_mine_2msgloss_atkAdaptMap_phase2_cap{args.seq_len}_wm.txt', 'r') as f:
        all_wm = []
        for line in f.readlines():
            if line.strip() != '':
                all_wm.append(line.strip())
    with open(f'test_0421_mine_2msgloss_atkAdaptMap_phase2_cap{args.seq_len}_orig.txt', 'r') as f:
        all_orig = f.readlines()
    for idx, (msgs, wm, orig) in enumerate(zip(all_msg, all_wm, all_orig)):
        # print(msgs)
        wm = wm.strip()
        orig = orig.strip()
        # print(wm)
        # print(orig)
        with torch.no_grad():
            # import pdb; pdb.set_trace()
            data_adv_autoenc = convert_words_to_idx(wm.split(' ')).cuda().unsqueeze(dim=1)
            data_dv_autoenc_emb = model_gen.forward_sent(data_adv_autoenc,msgs,args.gumbel_temp,only_embedding=True)
            msg_out = model_gen.forward_msg_decode(data_dv_autoenc_emb.detach())

			
            #get prediction (and the loss) of the discriminator on the fake sequence.
            sentences = [wm, orig]
            sbert_embs = sbert_model.encode(sentences)
            meteor_adv = meteor_score([orig.split(' ')],wm.split(' '))
            bert_diff_adv = np.linalg.norm(sbert_embs[0]-sbert_embs[1])
				
            msg_out = torch.round(sig(msg_out))
            msg_out = msg_out.view(-1)
            long_msg_out = torch.cat((long_msg_out, msg_out), dim=0)
            l2_distances = l2_distances + bert_diff_adv
            meteor_tot = meteor_tot + meteor_adv
            out_file.write('****'+'\n')
            out_file.write(str(batch_count)+'\n')
            meteor_pair = meteor_adv
            out_file.write(orig+'\n')

            out_file.write(str(bert_diff_adv) +'\n')				
            out_file.write(wm+'\n')

            #compute meteor score of the adv output
            out_file.write(str(compare_msg_bits(msgs,torch.round(sig(msg_out))))+'\n')
            out_file.write(np.array2string(msgs.detach().cpu().numpy().astype(int))+'\n')
            out_file.write(np.array2string(torch.round(sig(msg_out)).detach().cpu().numpy().astype(int))+'\n')
            # print(msgs)
            # print(torch.round(sig(msg_out)))
				
        batch_count = batch_count + 1

    # import pdb;pdb.set_trace()
    long_msg = long_msg[:long_msg_out.shape[0]]

    long_msg_count = 0
    tot_count = tot_count + 1
    tot_count_bits = tot_count_bits + long_msg.shape[0]
    long_msg_out = torch.round(sig(long_msg_out))
    similar_bits = compare_msg_bits(long_msg.unsqueeze(0),long_msg_out.unsqueeze(0))
    all_bits = long_msg.shape[0]
    correct_msg_count = correct_msg_count + compare_msg_whole(long_msg.unsqueeze(0),long_msg_out.unsqueeze(0))	
    correct_msg_count_bits = correct_msg_count_bits + similar_bits 
    p_value.append(binom_test(similar_bits, all_bits, 0.5))

    p_value_smaller = sum(i < 0.05 for i in p_value)
    Fscore = f1_score(y_label,y_out)
    # import pdb;pdb.set_trace()
    return total_loss_lm/batch_count, correct_msg_count/tot_count, (long_msg == long_msg_out).float().mean(), meteor_tot/batch_count, l2_distances/batch_count, fake_correct/batch_count, real_correct/batch_count, Fscore ,  np.mean(p_value), p_value_smaller/len(p_value)
	

	
	
	# Load the best saved model.
with open(args.gen_path, 'rb') as f:
    model_gen, _, _ , _= torch.load(f)
#print(model_gen)

with open(args.disc_path, 'rb') as f:
    model_disc, _, _ , _= torch.load(f)
#print(model_gen)

with open(args.autoenc_attack_path, 'rb') as f:
    model_autoenc_attack, _ , _= torch.load(f)
#print(model_gen)


if args.cuda:
    model_gen.cuda()
    model_disc.cuda()
    model_autoenc_attack.cuda()


# Run on test data.
f = open('test_out_0424.txt','w')
f_metrics = open('test_out_metrics_0424.txt','w')
test_lm_loss, test_correct_msg, test_correct_bits_msg, test_meteor, test_l2_sbert, test_correct_fake, test_correct_real, test_Fscore, test_pvalue, test_pvalue_inst = evaluate(test_data, f, test_batch_size)

print(test_lm_loss)
print(test_correct_msg)
print(test_correct_bits_msg)
print(test_meteor)
print(test_l2_sbert)
print(test_correct_fake)
print(test_correct_real)
print(test_Fscore)
print(test_pvalue)
print(test_pvalue_inst)

print('=' * 150)
print('| test | lm loss {:5.2f} | msg accuracy {:5.2f} | msg bit accuracy {:5.2f} | meteor {:5.4f} | SentBert dist. {:5.4f} | fake accuracy {:5.2f} | real accuracy {:5.2f} | F1 score {:.5f} | P-value {:.9f} | P-value inst {:5.2f}'.format(test_lm_loss, test_correct_msg*100, test_correct_bits_msg*100, test_meteor, test_l2_sbert, test_correct_fake*100,test_correct_real*100,test_Fscore,test_pvalue,test_pvalue_inst*100))
print('=' * 150)
f.close()

