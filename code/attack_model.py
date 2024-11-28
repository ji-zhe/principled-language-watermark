import torch
import torch.nn as nn
import math
import copy
import torch.nn.functional as F
from model_mt_autoenc_cce import PositionalEncoding, gumbel_softmax_sample

class Attacker(nn.Module):
    def __init__(self, ntoken, ninp, nlayers_encoder=6, transformer_drop=0.1, dropouti=0.15, dropoute=0.1, tie_weights=False, attention_heads=8, pretrained_model=None) -> None:
        super(Attacker, self).__init__()
        self.ninp = ninp
        self.embeddings = nn.Embedding(ntoken, ninp)
        self.sent_encoder_layer = nn.TransformerEncoderLayer(d_model=ninp, nhead=attention_heads, dropout=transformer_drop) 
        self.sent_dec_encoder_layer = nn.TransformerDecoderLayer(d_model=ninp, nhead=attention_heads, dropout=transformer_drop) 
        self.sent_encoder = nn.TransformerEncoder(self.sent_encoder_layer, nlayers_encoder)
        self.sent_decoder = nn.TransformerDecoder(self.sent_dec_encoder_layer, nlayers_encoder)
        self.pos_encoder = PositionalEncoding(ninp, transformer_drop)
        self.decoder = nn.Linear(ninp, ntoken)
        if pretrained_model:
            self.init_model(pretrained_model)

    def _generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask
    
    def init_model(self, pretrained_model):
        with torch.no_grad():
            self.embeddings.weight.data = copy.deepcopy(pretrained_model.embeddings.weight.data)
            self.sent_encoder = copy.deepcopy(pretrained_model.sent_encoder)

    def forward_transformer(self, input_data, gumbel_temp, one_hot=False):
        device = input_data.device
        mask = self._generate_square_subsequent_mask(len(input_data)).to(device)
        if one_hot:
            emb = torch.matmul(input_data, self.embeddings.weight)* math.sqrt(self.ninp)
        else:
            emb = self.embeddings(input_data) * math.sqrt(self.ninp)
        ### Encoder ###
        emb = self.pos_encoder(emb)
        input_data_emb = self.sent_encoder(emb)
        sent_embedding = input_data_emb.mean(dim=0)
        ### Decoder ###
        input_decoder = torch.zeros([input_data.size(0),input_data.size(1),self.ninp]).float()
        input_decoder = input_decoder.to(device)
        input_decoder[1:input_data_emb.size(0),:,:] = input_data_emb[0:input_data_emb.size(0)-1,:,:] # input_decoder = shifted input data emb
        sent_embeddings_repeat = sent_embedding.view(1,sent_embedding.size(0),sent_embedding.size(1)).repeat(input_data.size(0),1,1)  
        input_decoder = input_decoder + sent_embeddings_repeat # input_decoder = shifted input data emb + sentence embedding
        input_decoder = self.pos_encoder(input_decoder)

        sent_decoded = self.sent_decoder(input_decoder, memory=input_data_emb, tgt_mask=mask) # (S,N,E)
        sent_decoded_vocab = self.decoder(sent_decoded.view(sent_decoded.size(0)*sent_decoded.size(1), sent_decoded.size(2)))	# (S*N,V)
        sent_decoded_vocab_hot = F.gumbel_softmax(F.log_softmax(sent_decoded_vocab,dim=-1), tau = gumbel_temp, hard=True) # (S*N, V)
        sent_decoded_vocab_hot_out =  sent_decoded_vocab_hot.view(input_decoder.size(0), input_decoder.size(1), sent_decoded_vocab_hot.size(1)) # (S,N,V) one_hot

        sent_decoded_vocab_emb = torch.mm(sent_decoded_vocab_hot,self.embeddings.weight) #(S*N,E)
        sent_decoded_vocab_emb = sent_decoded_vocab_emb.view(input_decoder.size(0), input_decoder.size(1), input_decoder.size(2)) # (S,N,E)
		
        sent_decoded_vocab_soft = gumbel_softmax_sample(sent_decoded_vocab, tau = gumbel_temp)
		
        return sent_decoded_vocab_emb, sent_decoded_vocab_hot_out, sent_decoded_vocab_soft

class MI_seq2seq(nn.Module):
    def __init__(self, ntoken, ninp, transformer_drop, attention_heads, nlayers_encoder) -> None:
        super(MI_seq2seq,self).__init__()
        self.transformer_drop = transformer_drop
        self.pos_encoder = PositionalEncoding(ninp, transformer_drop)
        self.embd = nn.Linear(ntoken,ninp)
        self.sent_encoder_layer = nn.TransformerEncoderLayer(d_model=ninp, nhead=attention_heads, dropout=transformer_drop) 
        self.sent_encoder = nn.TransformerEncoder(self.sent_encoder_layer, nlayers_encoder)
        self.FC= nn.Linear(2*ninp,ninp)
        self.out = nn.Linear(ninp,1)
    
    def forward(self, sent_prob, wm_sent_prob):
        # NOTE: inputs are assumed batch_first (N,S,E) or (N,S,V)
        sent_emb = self.pos_encoder(self.embd(sent_prob.permute((1,0,2))))
        wm_sent_emb = self.pos_encoder(self.embd(wm_sent_prob.permute((1,0,2))))
        sent_feature = self.sent_encoder(sent_emb)
        wm_sent_feature = self.sent_encoder(wm_sent_emb) #(S,N,E)
        sent_feature = sent_feature.mean(dim=0) # (N,E)
        wm_sent_feature = wm_sent_feature.mean(dim=0)
        mix = self.FC(torch.cat((sent_feature, wm_sent_feature), dim=1))
        # import pdb;pdb.set_trace()
        # mix = self.FC(torch.cat((sent_emb.mean(dim=1), wm_sent_emb.mean(dim=1)) , dim=1))
        return self.out(mix).squeeze()

class MI_feature2seqAndBits(nn.Module):
    def __init__(self, ninp, transformer_drop, attention_heads, nlayers_encoder, key_len) -> None:
        super(MI_feature2seqAndBits,self).__init__()
        self.transformer_drop = transformer_drop
        self.sent_encoder_layer = nn.TransformerEncoderLayer(d_model=ninp, nhead=attention_heads, dropout=transformer_drop) 
        self.sent_encoder = nn.TransformerEncoder(self.sent_encoder_layer, nlayers_encoder)
        self.keyEmbed = nn.Linear(key_len, ninp)
        self.joinNet = nn.Linear(2*ninp, ninp) # join sequence and bits(key)
        self.FC= nn.Linear(2*ninp,ninp)
        self.out = nn.Linear(ninp,1)

    def forward(self,feature, seqAndBits):
        # import pdb;pdb.set_trace()
        seq, bits = seqAndBits
        seq_feature = self.sent_encoder(seq)
        seq_feature = seq_feature.mean(dim=0)
        bits_feature = self.keyEmbed(bits)
        seqAndBits_feature = self.joinNet(torch.cat((bits_feature,seq_feature),dim=1))
        if len(feature.shape) == len(seqAndBits_feature.shape)+1:
            feature = feature.mean(dim=0) # avg along seq
        feature = self.FC(torch.cat((seqAndBits_feature,feature),dim=1))
        return self.out(feature).squeeze()
        