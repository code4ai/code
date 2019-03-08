#!/usr/bin/env python2
# -*- coding: utf-8 -*-
import multiprocessing    
import numpy as np
np.random.seed(1337)  # for reproducibility
from gensim.models.word2vec import Word2Vec
from gensim.corpora.dictionary import Dictionary
import gensim

import config
conf = config.config()

def word2vec(data, use_bert=False, use_glove=False):
    print "word2vec training starting..."
    
    # set parameters:
    emb_dim = conf.emb_dim
    '''
    #build word2vec(skip-gram) model for dundee corpus
    model = Word2Vec(size=emb_dim,
                     min_count=0,
                     window=7,
                     workers=multiprocessing.cpu_count(),
                     iter=10,
                     sg=1)
                     
    model.build_vocab(data)
    model.train(data, 
                total_examples=model.corpus_count,
                epochs=model.iter)
    '''
    #model.wv.save_word2vec_format("./all_text_emb_%dd.bin"%conf.emb_dim, binary=True)
    model= gensim.models.KeyedVectors.load_word2vec_format("./all_text_emb_%dd.bin"%conf.emb_dim, binary=True)
    
    word_freq = []
    for w in model.vocab:
        word_freq.append([w.encode('utf-8'), model.vocab[w].count])
    word_freq = sorted(word_freq,key=lambda x:x[1], reverse=True)
    
    print "word2vec training is done!"
    gensim_dict = Dictionary()
    gensim_dict.doc2bow(model.wv.vocab.keys(),
                        allow_update=True)
    
    specials = ['PAD','END','BOS','UNK']+['UNK'+str(i) for i in range(1, conf.nb_unk+1)]
    
    word2id = {v: i+len(specials) for i, v in gensim_dict.items()}
    ###########################################################################
    if use_bert:
        n=0
        bert_word2vec=[]
        bert_word2vec_={}
        with open('bert-base-uncased.30522.768d.vec', 'r') as f:
            for e in f.readlines():
                bert_word2vec.append(e.split())
                
        for e in bert_word2vec[1:]:
            bert_word2vec_[e[0]]=np.array(e[1:]).astype('float32')
        bert_word2vec=bert_word2vec_
        bert_set=set(bert_word2vec.keys())
        
        word_vectors={}
        for word in word2id.keys():
            if word in bert_set:
                word_vectors[word]=bert_word2vec[word]
                n+=1
            else:
                word_vectors[word]=np.random.randn(emb_dim)
        print "%d words in bert found!!!"%n
    else:
        word_vectors = {word: model[word] for word in word2id.keys()}
    ###########################################################################
    for i, s in enumerate(specials):
        word2id[s]=i
        word_vectors[s]=np.random.randn(emb_dim)
    id2word = {word2id[num]: num for num in word2id.keys()}
    
    n_symbols = len(word2id)  # adding 1 to account for 0th index
    embedding_weights = np.zeros((n_symbols, emb_dim))
    for word, index in word2id.items():
        embedding_weights[index,:] = word_vectors[word]  
    
    if use_glove:
        print "glove embedding loading..."
        embedding_weights_=[]
        glo = gensim.models.KeyedVectors.load_word2vec_format("~/Desktop/all/desk/glove.6B/glove.6B.300d.txt", binary=False)
        for i in range(len(embedding_weights)):
            try:
                embedding_weights_.append(glo[id2word[i]])
            except:
                embedding_weights_.append(np.random.randn(300))
        embedding_weights = np.array(embedding_weights_)
        print "glove loading is down!"
        
    return word2id, id2word, word_vectors, embedding_weights, word_freq, len(specials)
