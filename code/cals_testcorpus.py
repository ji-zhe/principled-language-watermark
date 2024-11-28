import os
import hashlib
import sys
sys.path.append("..")
from utils import *
import data

fn = 'corpus.{}.data'.format(hashlib.md5("data/wikitext-2".encode()).hexdigest())
if os.path.exists(fn):
    print('Loading cached dataset...')
    corpus = torch.load(fn)
    # corpus = torch.jit.load(fn)
else:
    print('Producing dataset...')
    corpus = data.Corpus("data/wikitext-2")
    torch.save(corpus, fn)

eval_batch_size = 10
test_batch_size = 1

f = open("IF_rob/watermarked.txt")
g = open("IF_rob/orig.txt")

wm_texts = f.readlines()
orig_texts = g.readlines()

f.close()
g.close()
f = open("IF_rob/full_clean_watermarked.txt",'w')
g = open("IF_rob/full_clean_orig.txt",'w')

for wm, ori in zip(wm_texts, orig_texts):
    item_list = wm.strip('\n ').split('\t')
    wm_sent = item_list[-3]
    wm_words = wm_sent.split()
    clean_flag = True
    for word in wm_words:
        if word not in corpus.dictionary.word2idx:
            clean_flag = False
            print(word)
            break
    if clean_flag:
        f.write(wm)
        g.write(ori)
f.close()
g.close()
