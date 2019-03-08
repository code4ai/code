# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import numpy as np
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
USE_CUDA = torch.cuda.is_available()
FloatTensor = torch.cuda.FloatTensor if USE_CUDA else torch.FloatTensor
LongTensor = torch.cuda.LongTensor if USE_CUDA else torch.LongTensor
    
class EncoderLSTM(nn.Module):
    def __init__(self, pretrained_embedding_weights, 
                 vocab_size, 
                 emb_dim, 
                 hidden_dim, 
                 num_layers, 
		 word2id,
                 id2word,
		 word2unk,
                 word_vectors,
                 embedding_weights,
                 word_freq,
                 nb_specials,
                 batch_first=True, 
                 bidirectional=True):

        super(EncoderLSTM, self).__init__() 
        self.vocab_size = vocab_size
        self.emb_dim = emb_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.word2id = word2id
        self.id2word = id2word
        self.word2unk = word2unk
        self.word_vectors = word_vectors
        self.embedding_weights = embedding_weights
        self.word_freq = word_freq
        self.nb_specials = nb_specials
        self.num_directions = 2 if bidirectional else 1
        self.embed = nn.Embedding(self.vocab_size, self.emb_dim)
        self.embed.weight.data.copy_(torch.from_numpy(pretrained_embedding_weights))
        self.lstm = nn.LSTM(input_size=self.emb_dim, 
                           hidden_size=self.hidden_dim, 
                           num_layers=self.num_layers, 
                           batch_first=batch_first, 
                           bidirectional=bidirectional)
        #initialize linear
        self.params_init(self.lstm.named_parameters())
        
    def params_init(self, params):
        for name, param in params:
            if len(param.data.shape)==2:
                nn.init.kaiming_normal_(param, a=1, mode='fan_in')
            if len(param.data.shape)==1:
                nn.init.normal_(param)

    def forward(self, input_x, mask, hidden):
        input_x_embs= self.embed(input_x)        
        #sort the original seqs
        input_l = mask.sum(1)
        idx_sort = np.argsort(input_l)[::-1].tolist()
        idx_unsort = np.argsort(idx_sort).tolist()
        #packed the sorted seqs and avoid the impact of PAD on RNN model
        packed = torch.nn.utils.rnn.pack_padded_sequence(input_x_embs[idx_sort], \
                                                         list(input_l[idx_sort]), batch_first=True)
        output, (hn, cn) = self.lstm(packed, hidden)
        #hn: [num_layers * num_directions, batch, hidden_size]
        hx, _ = torch.nn.utils.rnn.pad_packed_sequence(output, batch_first=True)
        hx = hx[idx_unsort]
        hn = hn[:,idx_unsort,:]
        cn = cn[:,idx_unsort,:]
        return hx, (hn, cn) #torch.mean(hx, dim=1)
    
    def initHidden(self, batch_size):
        h = torch.zeros(self.num_layers*self.num_directions, batch_size, self.hidden_dim, device=device)
        c = torch.zeros(self.num_layers*self.num_directions, batch_size, self.hidden_dim, device=device)
        return (h, c)
          
