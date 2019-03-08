# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import numpy as np
from Attention import Attention

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
USE_CUDA = torch.cuda.is_available()
FloatTensor = torch.cuda.FloatTensor if USE_CUDA else torch.FloatTensor
LongTensor = torch.cuda.LongTensor if USE_CUDA else torch.LongTensor

class DecoderLSTM(nn.Module):
    def __init__(self, pretrained_embedding_weights, 
                 vocab_size, emb_dim, dec_hdim, num_layers, batch_size,
                 batch_first=True, 
                 bidirectional=False,
                 attn_type="general"):
        super(DecoderLSTM, self).__init__() 
        self.vocab_size = vocab_size
        self.emb_dim = emb_dim
        self.dec_hdim = dec_hdim
        self.num_layers = num_layers
        self.batch_size = batch_size
        self.attn_type = attn_type
        self.embed = nn.Embedding(self.vocab_size, self.emb_dim)
        self.embed.weight.data.copy_(torch.from_numpy(pretrained_embedding_weights))
        self.linear = nn.Linear(self.dec_hdim*2, self.vocab_size, bias=False)#*2
        self.lstm = nn.LSTM(input_size=self.emb_dim, 
                          hidden_size=self.dec_hdim, 
                          num_layers=self.num_layers, 
                          batch_first=batch_first, 
                          bidirectional=False)
        
        self.sm = nn.Softmax(dim=-1)
        self.attn = Attention(emb_dim, dec_hdim, attn_type)
        self.params_init(self.linear.named_parameters())
        self.params_init(self.lstm.named_parameters())
    
    def params_init(self, params):
        for name, param in params:
            if len(param.data.shape)==2:
                nn.init.kaiming_normal_(param, a=1, mode='fan_in')
            if len(param.data.shape)==1:
                nn.init.normal_(param)

    def forward(self, 
                hx_src, 
                tgt_seqs, #numpy
                mask_src,
                mask_tgt,
                decoder_iniHidden):
        #h0: [4, 32, 128]
        h0, c0 = decoder_iniHidden
        #hn [num_layers, num_directions, batch, hidden_size]
        h0=h0.view(self.num_layers, 2, self.batch_size, -1)
        h0 = torch.cat([h0[:,0,:,:], h0[:,1,:,:]], dim=-1)
        
        c0=h0.view(self.num_layers, 2, self.batch_size, -1)
        c0 = torch.cat([c0[:,0,:,:], c0[:,1,:,:]], dim=-1)
        
        bos = np.array([len(tgt_seqs)*[3]])
        tgt_seqs_input = np.insert(tgt_seqs[:,:-1], 0, bos, axis=1)
        tgt_seqs_input = self.embed(LongTensor(tgt_seqs_input))
        #sort the original seqs
        tgt_lens = mask_tgt.sum(1)
        idx_sort = np.argsort(tgt_lens)[::-1].tolist()
        idx_unsort = np.argsort(idx_sort).tolist()
        #packed the sorted seqs and avoid the impact of PAD on RNN model
        packed = torch.nn.utils.rnn.pack_padded_sequence(tgt_seqs_input[idx_sort], \
                                                 list(tgt_lens[idx_sort]), batch_first=True)
    
        output, _ = self.lstm(packed, (h0[:, idx_sort, :], c0[:, idx_sort, :]))
        st, _ = torch.nn.utils.rnn.pad_packed_sequence(output, batch_first=True)
        st = st[idx_unsort]
        
        ct_list=[]
        for i in range(st.shape[1]):
            ct_list.append(self.attn(st[:,i,:], hx_src, mask_src, None))
        ct_list = [ct.unsqueeze(1) for ct in ct_list]
        ct = torch.cat(ct_list, dim=1)
        mask_tgt = FloatTensor(mask_tgt).unsqueeze(-1)
        concat_h = torch.cat([st, ct], dim=2)*mask_tgt
        
        logits = self.linear(concat_h)
        pred = logits.argmax(2)
        return logits, tgt_seqs, tgt_lens, pred


