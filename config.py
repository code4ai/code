#/usr/bin/env python2
# -*- coding: utf-8 -*-

class config(object):
   
    emb_dim        = 768
    hidden_dim     = 512 #encode_hdim=hidden_dim; decode_hdim=hidden_dim*2
    num_layers     = 3
    num_directions = 3
    vocab_size     = 50000
    enc_vocab_size = vocab_size #including 4 special tokens
    dec_vocab_size = vocab_size
    batch_size     = 8
    nb_unk         = 20
    beam           = 1
   
    lamda          = 0.1
    lr             = 0.0002
    decay_rate     = 0.99
    gamma          = 0.99
    
    
