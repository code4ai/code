# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import config
conf = config.config()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
USE_CUDA = torch.cuda.is_available()
FloatTensor = torch.cuda.FloatTensor if USE_CUDA else torch.FloatTensor
LongTensor = torch.cuda.LongTensor if USE_CUDA else torch.LongTensor

class Attention(nn.Module):
    #attn_type = ["dot", "bilinear", "prior"] 
    def __init__(self, emb_dim, dec_hdim, attn_type="general"):
        super(Attention, self).__init__() 
        self.attn_type = attn_type
        self.emb_dim = emb_dim
        self.dec_hdim = dec_hdim
        self.w_bilinear = nn.Linear(self.dec_hdim, self.dec_hdim)
        self.params_init(self.w_bilinear.named_parameters())
        self.sm = nn.Softmax(dim=-1)
    
    def params_init(self, params):
        for name, param in params:
            if len(param.data.shape)==2:
                nn.init.kaiming_normal_(param, a=1, mode='fan_in')
            if len(param.data.shape)==1:
                nn.init.normal_(param)

    def forward(self, st, hx, hx_mask, prior_att=None):
        hx_mask1 = FloatTensor(hx_mask+(1-hx_mask)*1e-18)
        src_batch, src_len, src_dim = hx.size()
        st = torch.unsqueeze(st, -1)
        score = prior_att
        if score==None:
            score=torch.ones(src_batch, src_len, 1)
            if self.attn_type=="dot":
                score = torch.bmm(hx, st).squeeze(-1)
                score = score*hx_mask1
                score = self.sm(score).unsqueeze(-1)
                
            if self.attn_type=="general":
                hx_ = hx.contiguous().view(src_batch*src_len, src_dim)
                hx_ = self.w_bilinear(hx_)
                hx = hx_.contiguous().view(src_batch, src_len, src_dim)
                score = torch.bmm(hx, st).squeeze(-1)
                score = score*hx_mask1
                score = self.sm(score).unsqueeze(-1)
        ct = torch.sum(score*hx, dim=-2)
        return ct
        
