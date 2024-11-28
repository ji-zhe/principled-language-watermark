import argparse
import numpy as np
from sentence_transformers import SentenceTransformer
from nltk.translate.meteor_score import meteor_score
from transformers import AutoTokenizer
import spacy

import argparse

parser = argparse.ArgumentParser(description='Evaluate transformer')

parser.add_argument('--ori', type=str, default='',
                    help='location of read')

parser.add_argument('--cor', type=str, default='',
                    help='location of read')
					
args = parser.parse_args()

'''
CUDA_VISIBLE_DEVICES=7 nohup python -u cals_compare.py --ori cals/wt2/full_clean_orig.txt --corrupted cals/wt2/corrupted__clean.txt > evaluate_bertemb.log &
CUDA_VISIBLE_DEVICES=7 nohup python -u  --origin cals/full_clean_orig.txt --corrupted cals/corrupted_bertembunlock_clean.txt > evaluate_bertembunlock.log &
'''

seed = 1111
samples_num = 10
gumbel_temp = 0.5
tokenizer=AutoTokenizer.from_pretrained('bert-base-cased')

spacynlp=spacy.load("en_core_web_sm")
sbert_model = SentenceTransformer('bert-base-nli-mean-tokens') 

meteor_list = []
bert_diff_list = []
spacy_meteor_list = []
with open(args.cor,'r',encoding='utf-8') as fp1, open(args.ori,'r',encoding='utf-8') as fp2:
    for orig,corru in zip(fp1,fp2):
        sentences = [corru,orig]
        sbert_embs = sbert_model.encode(sentences)
        bert_diff_adv = np.linalg.norm(sbert_embs[0]-sbert_embs[1])
        
        meteor_adv = meteor_score([tokenizer.tokenize(orig)],tokenizer.tokenize(corru))
        spacy_meteor_adv = meteor_score([[token.text for token in spacynlp(orig)]],
                                        [token.text for token in spacynlp(corru)])
        
        meteor_list.append(meteor_adv)
        spacy_meteor_list.append(spacy_meteor_adv)
        bert_diff_list.append(bert_diff_adv)
        print('meteor: {},{}, sbert: {}'.format(meteor_adv,spacy_meteor_adv,bert_diff_adv))
        
print('avg meteor',np.mean(meteor_list),np.mean(spacy_meteor_list))
print('avg sbert',np.mean(bert_diff_list))

            
