import argparse
import numpy as np
import torch
import torch.nn as nn
import spacy
from sentence_transformers import SentenceTransformer
from nltk.translate.meteor_score import meteor_score

from transformers import AutoTokenizer

import argparse

parser = argparse.ArgumentParser(description='Attack texts use DAE')

parser.add_argument('--model', type=str, default='',
                    help='location of model')

parser.add_argument('--origin', type=str, default='',
                    help='location of read')

parser.add_argument('--watermarked', type=str, default='',
                    help='location of read')

parser.add_argument('--clean', type=str, default='',
                    help='location of clean write')

parser.add_argument('--composite', type=str, default='',
                    help='location of composite write')
					
args = parser.parse_args()

gumbel_temp = 0.5
seed = 1111

sbert_model = SentenceTransformer('bert-base-nli-mean-tokens') 
spacynlp = spacy.load("en_core_web_sm") 
tokenizer=AutoTokenizer.from_pretrained("bert-base-cased")

def get_idx_from_logits(sequence,seq_len,bsz):
    m = nn.Softmax(dim=-1)
    sequence = sequence.view(seq_len,bsz,sequence.size(1))
    sequence = m(sequence)    
    sequence_idx = torch.argmax(sequence, dim=-1)
    return sequence_idx

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

with open(args.model, 'rb') as f:
    model_autoenc_attack, _ , _= torch.load(f)
    model_autoenc_attack.cuda()

with open(args.watermarked, 'r') as readf, open(args.origin, 'r') as reado, open(args.clean,'w') as writeclean, open(args.composite,'w') as writecomposite:
    bert_diff_list = []
    bert_meteor_list = []
    spacy_meteor_list = []
    for line, orig_text in zip(readf,reado):
        line=line.split('\t')
        replaced_line=line[4]
        token=torch.LongTensor(tokenizer(replaced_line)['input_ids'][1:-1]).cuda()
        # In this, we remove both [CLS] and [SEP]
        token = token.view(-1,1)
        auc=autoenc_greedy(token,1)
        output_text = tokenizer.decode(auc[:-1,0])
        
        writeclean.write(output_text+'\n')
        line[4]=output_text
        writecomposite.write("\t".join(line))
        
        sentences = [output_text,orig_text]
        sbert_embs = sbert_model.encode(sentences)
        bert_diff_adv = np.linalg.norm(sbert_embs[0]-sbert_embs[1])
        bert_diff_list.append(bert_diff_adv)
        
        bert_meteor_adv = meteor_score([tokenizer.tokenize(orig_text)],tokenizer.tokenize(output_text))
        bert_meteor_list.append(bert_meteor_adv)
        
        spacy_meteor_adv = meteor_score([[token.text for token in spacynlp(orig_text)]],
                                        [token.text for token in spacynlp(output_text)])
        spacy_meteor_list.append(spacy_meteor_adv)
        
        print(f'bert meteor: {bert_meteor_adv}, spacy meteor: {spacy_meteor_adv}, sbert: {bert_diff_adv}')
        
    print('avg sbert',np.mean(bert_diff_list))
    print('avg bert meteor',np.mean(bert_meteor_list))
    print('avg spacy meteor',np.mean(spacy_meteor_list))
