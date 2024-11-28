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

data_path = 'data/wikitext-2'
seed = 1111
autoenc_attack_path = "model1_denoise_autoenc_attack.pt"
samples_num = 10
gumbel_temp = 0.5

sbert_model = SentenceTransformer('bert-base-nli-mean-tokens') 
with open(autoenc_attack_path, 'rb') as f:
    model_autoenc_attack, _ , _= torch.load(f)

model_autoenc_attack.cuda()

f = open('./IF_rob/full_clean_watermarked.txt')
wm_sents = f.readlines()
f.close()

import os
import hashlib
fn = 'corpus.{}.data'.format(hashlib.md5(data_path.encode()).hexdigest())
if os.path.exists(fn):
    print('Loading cached dataset...')
    corpus = torch.load(fn)
    # corpus = torch.jit.load(fn)
else:
    print('Producing dataset...')
    corpus = data.Corpus(data_path)
    torch.save(corpus, fn)

def random_cut_sequence(sequence, limit=10):

    rand_cut_start = np.random.randint(low=0, high=limit)
    rand_cut_end = sequence.size(0) - np.random.randint(low=0, high=limit)

    sequence = sequence[rand_cut_start:rand_cut_end, :]
    new_seq_len = rand_cut_end -  rand_cut_start 
    #print(sequence.size())
    return sequence

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

def convert_idx_to_words(idx):
    batch_list = []
    for i in range(0,idx.size(1)):
        sent_list = []
        for j in range(0,idx.size(0)):
            sent_list.append(corpus.dictionary.idx2word[idx[j,i]])
        batch_list.append(sent_list)
    return batch_list


def autoenc_greedy(data, batch_size):
    sent_emb, encoder_out = model_autoenc_attack.forward_sent_encoder(data)
    sent_out_soft = model_autoenc_attack.forward_sent_decoder(sent_emb, data, encoder_out, gumbel_temp)
    sent_out = get_idx_from_logits(sent_out_soft,data.size(0),batch_size)
    sent_out = sent_out[0,:]
    sent_out = torch.cat( (sent_out, torch.zeros(1,dtype=torch.long).cuda()), axis=0)
    sent_out = sent_out.view(2,1)
	
    for j in range(1,data.size(0)):
        sent_out_soft =  model_autoenc_attack.forward_sent_decoder(sent_emb, sent_out, encoder_out, gumbel_temp)
        sent_out_new = get_idx_from_logits(sent_out_soft,j+1,batch_size)				
        sent_out = torch.cat( (sent_out[0:j,:].view(j,1),sent_out_new[j,:].view(1,1)),axis=0)
        sent_out = torch.cat( (sent_out, torch.zeros( (1,1),dtype=torch.long).cuda()), axis=0)
    return sent_out

g = open('IF_rob/full_clean_orig.txt')
h = open('IF_rob/orig_in_corrupt_0626.txt','w')
orig_sents = g.readlines()
print("corrupted v.s. clean watermarked")
print("meteor, sbert", "corrupted to orig")
with open('IF_rob/tmp_corrupted.txt','w') as f:
    meteor_list = []
    bert_diff_list = []
    for wm_sent_item, orig in zip(wm_sents, orig_sents):
        # try:
            item_list = wm_sent_item.strip('\n ').split('\t')
            wm_sent = item_list[-3]
            sent = []
            for word in wm_sent.split():
                sent.append(corpus.dictionary.word2idx[word])
            sent = torch.tensor(sent)
            sent = sent.cuda()
            sent = sent.view(-1,1)
            data_adv_autoenc = autoenc_greedy(sent,1)

            adv_sent = convert_idx_to_words(data_adv_autoenc)
            adv_sent = adv_sent[0]
            adv_sent = ' '.join(adv_sent)

            sentences = [adv_sent, orig]
            sbert_embs = sbert_model.encode(sentences)
            meteor_adv = meteor_score([orig.split(' ')],adv_sent.split(' '))
            bert_diff_adv = np.linalg.norm(sbert_embs[0]-sbert_embs[1])
            meteor_list.append(meteor_adv)
            bert_diff_list.append(bert_diff_adv)


            # print(adv_sent)
            item_list[-3] = adv_sent
            f.write('\t'.join(item_list)+'\n')
            h.write(orig)
        # except Exception as e:
        #     ccc = wm_sent_item
        #     import pdb; pdb.set_trace()
    print('meteor',np.mean(meteor_list))
    print('sbert',np.mean(bert_diff_list))

            
