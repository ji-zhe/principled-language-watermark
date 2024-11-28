import torch
import torch.nn as nn
import math

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.2, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)

class MI_seq2seq(nn.Module):
    def __init__(self, ninp, transformer_drop, attention_heads, nlayers_encoder, pos_drop=0.2) -> None:
        super(MI_seq2seq,self).__init__()
        self.transformer_drop = transformer_drop
        self.pos_encoder = PositionalEncoding(ninp, pos_drop)
        self.sent_encoder_layer = nn.TransformerEncoderLayer(d_model=ninp, nhead=attention_heads, dropout=transformer_drop) 
        self.sent_encoder = nn.TransformerEncoder(self.sent_encoder_layer, nlayers_encoder)
        self.FC= nn.Sequential(
            nn.Linear(2*ninp,2*ninp),
            nn.LeakyReLU(),
            nn.Linear(2*ninp,ninp),
            nn.LeakyReLU()
        )
        self.out = nn.Linear(ninp,1)
    
    def forward(self, sent, wm_sent):
        ##### inputs are batch first! ####
        sent = self.pos_encoder(sent.permute((1,0,2)))
        sent_feature = self.sent_encoder(sent)
        wm_sent = self.pos_encoder(wm_sent.permute((1,0,2)))
        wm_sent_feature = self.sent_encoder(wm_sent) #(S,N,E)
        sent_feature = sent_feature.mean(dim=0)
        wm_sent_feature = wm_sent_feature.mean(dim=0)
        mix = self.FC(torch.cat((sent_feature, wm_sent_feature),dim=-1))
        return self.out(mix).squeeze()

class MI_feature2seq(nn.Module):
    def __init__(self, ninp, transformer_drop, attention_heads, nlayers_encoder, pos_drop=0.2) -> None:
        super(MI_feature2seq,self).__init__()
        self.transformer_drop = transformer_drop
        self.pos_encoder = PositionalEncoding(ninp, pos_drop)
        self.sent_encoder_layer = nn.TransformerEncoderLayer(d_model=ninp, nhead=attention_heads, dropout=transformer_drop) 
        self.sent_encoder = nn.TransformerEncoder(self.sent_encoder_layer, nlayers_encoder)
        self.FC= nn.Sequential(
            nn.Linear(2*ninp,2*ninp),
            nn.LeakyReLU(),
            nn.Linear(2*ninp,ninp),
            nn.LeakyReLU()
        )
        self.out = nn.Linear(ninp,1)
    
    def forward(self, feat, sent):
        ##### inputs are batch first! ####
        sent = self.pos_encoder(sent.permute((1,0,2)))
        sent_feature = self.sent_encoder(sent) # (80 SEQ, 80 BATCH, 512 EMB)
        sent_feature = sent_feature.mean(dim=0) # (80BATCH, 512)
        mix = self.FC(torch.cat((feat, sent_feature),dim=-1))
        return self.out(mix).squeeze()
    
class MI_feature2seq_key(nn.Module):
    def __init__(self, ninp, transformer_drop, attention_heads, nlayers_encoder, pos_drop=0.2) -> None:
        super(MI_feature2seq_key,self).__init__()
        self.transformer_drop = transformer_drop
        self.pos_encoder = PositionalEncoding(ninp, pos_drop)
        self.sent_encoder_layer = nn.TransformerEncoderLayer(d_model=ninp, nhead=attention_heads, dropout=transformer_drop) 
        self.sent_encoder = nn.TransformerEncoder(self.sent_encoder_layer, nlayers_encoder)
        self.FC= nn.Sequential(
            nn.Linear(3*ninp,2*ninp),
            nn.LeakyReLU(),
            nn.Linear(2*ninp,ninp),
            nn.LeakyReLU()
        )
        self.out = nn.Linear(ninp,1)
    
    def forward(self, feat, sent, key_sent):
        ##### inputs are batch first! ####
        sent = self.pos_encoder(sent.permute((1,0,2)))
        sent_feature = self.sent_encoder(sent) # (80 SEQ, 80 BATCH, 512 EMB)
        sent_feature = sent_feature.mean(dim=0) # (80BATCH, 512)
        key_sent = self.pos_encoder(key_sent.permute((1,0,2)))
        key_sent_feature = self.sent_encoder(key_sent) # (80 SEQ, 80 BATCH, 512 EMB)
        key_sent_feature = key_sent_feature.mean(dim=0) # (80BATCH, 512)
        mix = self.FC(torch.cat((feat, sent_feature, key_sent_feature),dim=-1))
        # mix = self.FC(torch.cat((feat, sent_feature),dim=-1))
        return self.out(mix).squeeze()

class MI_feature2seq_keydiff(nn.Module):
    def __init__(self, ninp, transformer_drop, attention_heads, nlayers_encoder, pos_drop=0.2) -> None:
        super(MI_feature2seq_keydiff,self).__init__()
        self.transformer_drop = transformer_drop
        self.pos_encoder = PositionalEncoding(ninp, pos_drop)
        self.sent_encoder_layer = nn.TransformerEncoderLayer(d_model=ninp, nhead=attention_heads, dropout=transformer_drop) 
        self.sent_encoder = nn.TransformerEncoder(self.sent_encoder_layer, nlayers_encoder)
        self.FC= nn.Sequential(
            nn.Linear(2*ninp,ninp),
            nn.LeakyReLU()
        )
        self.fc = nn.Linear(2*ninp, ninp)
        self.out = nn.Linear(ninp,1)
    
    def forward(self, feat, sent, keys):
        ##### inputs are batch first! ####
        sent = self.pos_encoder(sent.permute((1,0,2)))
        orig = sent.clone()
        # import pdb;pdb.set_trace()
        orig[keys.mask] = keys.emb[keys.mask]
        input = torch.cat([sent,orig], dim=2)
        input = self.fc(input)
        sent_feature = self.sent_encoder(sent) # (80 SEQ, 80 BATCH, 512 EMB)
        sent_feature = sent_feature.mean(dim=0) # (80BATCH, 512)
        mix = self.FC(torch.cat((feat, sent_feature),dim=-1))
        return self.out(mix).squeeze()

class MI_seqFeat2seqProb_dense(nn.Module):
    def __init__(self, ninp, ntoken, transformer_drop, attention_heads, nlayers_encoder, pos_drop=0.2, seq_len=80) -> None:
        super(MI_seqFeat2seqProb_dense,self).__init__()
        self.embeddings = nn.Embedding(ntoken,ninp)
        self.transformer_drop = transformer_drop
        self.pos_encoder = PositionalEncoding(ninp, pos_drop)
        self.sent_encoder_layer = nn.TransformerEncoderLayer(d_model=ninp, nhead=attention_heads, dropout=transformer_drop) 
        self.sent_encoder = nn.TransformerEncoder(self.sent_encoder_layer, nlayers_encoder)
        self.dense = nn.Linear(seq_len*ninp, ninp)
        self.FC= nn.Sequential(
            nn.Linear(2*ninp,2*ninp),
            nn.LeakyReLU(),
            nn.Linear(2*ninp,ninp),
            nn.LeakyReLU()
        )
        self.out = nn.Linear(ninp,1)
    def forward(self, feat, prob, skip1=False, skip2=False):
        # batch first
        prob_emb = torch.matmul(prob, self.embeddings.weight)
        prob_emb = prob_emb.permute((1,0,2))
        prob_emb = self.pos_encoder(prob_emb)
        feat2 = self.sent_encoder(prob_emb).permute((1,0,2)) # batch first
        feat2 = self.dense(feat2.reshape((feat2.shape[0],-1)))
        feat = self.dense(feat.reshape((feat.shape[0],-1)))
        mix = self.FC(torch.cat((feat, feat2),dim=-1))
        return self.out(mix).squeeze()
    
class MI_3seq(nn.Module):
    def __init__(self, ninp, transformer_drop, attention_heads, nlayers_encoder, pos_drop=0.2) -> None:
        super(MI_3seq,self).__init__()
        self.transformer_drop = transformer_drop
        self.pos_encoder = PositionalEncoding(ninp, pos_drop)
        self.sent_encoder_layer = nn.TransformerEncoderLayer(d_model=ninp, nhead=attention_heads, dropout=transformer_drop) 
        self.sent_encoder = nn.TransformerEncoder(self.sent_encoder_layer, nlayers_encoder)
        self.FC= nn.Sequential(
            nn.Linear(3*ninp,2*ninp),
            nn.LeakyReLU(),
            nn.Linear(2*ninp,ninp),
            nn.LeakyReLU()
        )
        self.out = nn.Linear(ninp,1)
    
    def forward(self, sent, wm_sent, key_sent):
        ##### inputs are batch first! ####
        sent = self.pos_encoder(sent.permute((1,0,2)))
        sent_feature = self.sent_encoder(sent)
        wm_sent = self.pos_encoder(wm_sent.permute((1,0,2)))
        wm_sent_feature = self.sent_encoder(wm_sent) #(S,N,E)
        key_sent = self.pos_encoder(key_sent.permute((1,0,2)))
        key_sent_feature = self.sent_encoder(key_sent) #(S,N,E)
        sent_feature = sent_feature.mean(dim=0)
        wm_sent_feature = wm_sent_feature.mean(dim=0)
        key_sent_feature = key_sent_feature.mean(dim=0)
        mix = self.FC(torch.cat((sent_feature, wm_sent_feature, key_sent_feature),dim=-1))
        return self.out(mix).squeeze()
    
class MI_3feat(nn.Module):
    def __init__(self, ninp, transformer_drop, attention_heads, nlayers_encoder, pos_drop=0.2) -> None:
        super(MI_3feat,self).__init__()
        self.FC= nn.Sequential(
            nn.Linear(3*ninp,2*ninp),
            nn.LeakyReLU(),
            nn.Linear(2*ninp,ninp),
            nn.LeakyReLU()
        )
        self.out = nn.Linear(ninp,1)
    
    def forward(self, sent_feature, wm_sent_feature, key_sent_feature):
        ##### inputs are batch first! ####
        mix = self.FC(torch.cat((sent_feature, wm_sent_feature, key_sent_feature),dim=-1))
        return self.out(mix).squeeze()
    
class MI_2feat(nn.Module):
    def __init__(self, ninp, transformer_drop, attention_heads, nlayers_encoder, pos_drop=0.2) -> None:
        super(MI_2feat,self).__init__()
        self.FC= nn.Sequential(
            nn.Linear(2*ninp,2*ninp),
            nn.LeakyReLU(),
            nn.Linear(2*ninp,ninp),
            nn.LeakyReLU()
        )
        self.out = nn.Linear(ninp,1)
    
    def forward(self, sent_feature, wm_sent_feature):
        ##### inputs are batch first! ####
        mix = self.FC(torch.cat((sent_feature, wm_sent_feature),dim=-1))
        return self.out(mix).squeeze()