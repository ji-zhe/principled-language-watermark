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

data = []

file = open('watermarked.txt', 'r')

file_data = file.readlines()

for row in file_data:
    data.append(row)

with open('model1_denoise_autoenc_attack.pt', 'rb') as f:
    model_autoenc_attack, _ , _= torch.load(f)

model_autoenc_attack.cuda()

word_idx_beams = []


sent_emb, enc_out = model_autoenc_attack.forward_sent_encoder(data[0])

print(sent_emb)
