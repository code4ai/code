# -*- coding: utf-8 -*-
from Encoder import EncoderLSTM 
from Decoder import DecoderLSTM 
import numpy as np
import config, Word2vec1, torch, argparse
import torch.nn.functional as F
from masked_cross_entropy import compute_loss
conf = config.config()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
USE_CUDA = torch.cuda.is_available()
FloatTensor = torch.cuda.FloatTensor if USE_CUDA else torch.FloatTensor
LongTensor = torch.cuda.LongTensor if USE_CUDA else torch.LongTensor
#%%
def show(data, string):
    print "****************"
    print string, data.shape
    print "\n"
    print data
    print "****************"
    
def invert_dict(d):
    return dict((v,k) for k,v in d.iteritems())

def read(path):
    data=[]
    with open(path, 'r') as f:
        for line in f.readlines():
            data.append(line.split())
    return data

def data_preparetion(input_txt, output_txt, limit_x, limit_y):
    data_x=[]
    data_y=[]
    with open(input_txt, 'r') as f:
        for line in f.readlines():
            data_x.append(line.split()[:limit_x])
    with open(output_txt, 'r') as f:
        for line in f.readlines():
            data_y.append(line.split())
    
    data_x1=[]
    data_y1=[]
    for x ,y in zip(data_x, data_y):
        if len(x)<=limit_x and len(y)<=limit_y:
            data_x1.append(x)
            data_y1.append(y)
    
    print "The final data amount is %d"%len(data_x1)
    data_x = data_x1
    data_y = data_y1
    return data_x, data_y

def unk_token(word2id, id2word, word_freq, word_vectors):
    word2id_ = {}
    embedding_weights=[]
    word_freq = [['PAD',1], ['END',1], ['BOS',1], ['UNK', 1]]+[['UNK'+str(i), 1] for i in range(1, conf.nb_unk+1)]+word_freq
    
    for i in range(len(word_freq)):
        word2id_[word_freq[i][0]] = i
        embedding_weights.append(word_vectors[id2word[i]])
    
    id2word_ = {word2id_[num]: num for num in word2id_.keys()}
    return word2id_, id2word_, np.array(embedding_weights)

def source2id(data, word2id):
    data_id=[]
    data_unk=[]
    vocab_set=set(word2id.keys())
    for sent in data:
        temp=[]
        word2unk={}
        numbering=1
        for w in sent:
            if w in vocab_set:
                temp.append(word2id[w])
            else:
                if w not in word2unk.keys():
                    word2unk[w]='UNK'+str(numbering)
                    #print word2unk
                    numbering+=1
                    
                    try:
                        temp.append(word2id[word2unk[w]])
                    except:
                        print "error!"
                        print word2unk
                        print " ".join(sent)
                        print "##########################"
                else:
                    temp.append(word2id[word2unk[w]])
                    
        data_unk.append(word2unk)
        data_id.append(temp)
    return data_id, data_unk

def target2id(data, word2id, data_unk):
    data_id=[]
    for sent, word2unk in zip(data, data_unk):
        temp=[]
        unk_set=set(word2unk.keys())
        vocab_set=set(word2id.keys())
        if unk_set=={}:
            for w in sent:
                if w in vocab_set:
                    temp.append(word2id[w])
                else:
                    temp.append(word2id['UNK'])
        else:
            for w in sent:
                if w in unk_set:
                    temp.append(word2id[word2unk[w]])
                else:
                    if w in vocab_set:
                        temp.append(word2id[w])
                    else:
                        temp.append(word2id['UNK'])
        
        data_id.append(temp)
    return data_id
        
def batch_padded(seq_ids):
    padded_seq_ids=[]
    mask=[]
    max_len = max([len(e) for e in seq_ids])
    for e in seq_ids:
        padded_seq_ids.append(e+(max_len-len(e))*[0])
        mask.append(len(e)*[1]+(max_len-len(e))*[0])
    return np.array(padded_seq_ids), np.array(mask)

def args_init():
    parser=argparse.ArgumentParser()
    parser.add_argument('--input'.encode('utf-8'), type=str, default='source.txt')
    parser.add_argument('--output'.encode('utf-8'), type=str, default='target.txt')
    return parser.parse_args()

#%%
MAX_LENGTH = 40
def evaluate(bz, encoder, decoder, val_data_unk, all_val_scr_seqs, all_val_tgt_seqs, id2word_np, beam_width, max_length=MAX_LENGTH):
    res=[]
    with torch.no_grad():
        nb = len(all_val_scr_seqs)/bz
        for b_i in range(nb):
            val_scr_seqs, val_scr_mask = batch_padded(all_val_scr_seqs[b_i*bz:(b_i+1)*bz])
            val_tgt_seqs, val_tgt_mask = batch_padded(all_val_tgt_seqs[b_i*bz:(b_i+1)*bz])
            
            encoder_h0 = encoder.initHidden(bz)
            hx_src, decoder_iniHidden = encoder(LongTensor(val_scr_seqs), val_scr_mask, encoder_h0)
            input_t = torch.zeros(bz, 1, conf.emb_dim, device=device)
            h0, c0 = decoder_iniHidden
            #hn [num_layers, num_directions, batch, hidden_size]
            h0=h0.view(conf.num_layers, 2, bz, -1)
            h0 = torch.cat([h0[:,0,:,:], h0[:,1,:,:]], dim=-1)
          
            c0=h0.view(conf.num_layers, 2, bz, -1)
            c0 = torch.cat([c0[:,0,:,:], c0[:,1,:,:]], dim=-1)
            if beam_width==1:
                batch_pred=[]
                for step in range(max_length):
                    st, (h0, c0) = decoder.lstm(input_t, (h0, c0))
                    st = st.squeeze(1)
                    ct = decoder.attn(st, hx_src, val_scr_mask)
                    concat_h = torch.cat([st, ct], dim=1)
                    logits = decoder.linear(concat_h)
                    pred = logits.argmax(1)
                    
                    input_t = decoder.embed(pred).unsqueeze(1)
                    batch_pred.append(pred.cpu().numpy())
                  
                batch_pred = np.array(batch_pred)
                res.extend(batch_pred.T)
                print "finished %d sentences"%(b_i*bz)
            else:
                candis = [[[], 0.0, 0] for _ in range(beam_width)]
                beam_hidden=[(h0, c0) for _ in range(beam_width)]
		batch_pred=[]
                for step in range(max_length):
                    all_candidates = []
                    for i in range(len(candis)):
                        seq, score, beam_i = candis[i]
                        if seq==[]:
                            input_t = torch.zeros(bz, 1, conf.emb_dim, device=device)
                        else:
                            pred = LongTensor(np.array([seq[-1]]))
                            input_t = decoder.embed(pred).unsqueeze(1)
                        st, (h0, c0) = decoder.lstm(input_t, beam_hidden[beam_i])
                        beam_hidden[i] = (h0, c0)
			st = st.squeeze(1)
                        ct = decoder.attn(st, hx_src, val_scr_mask)
                        concat_h = torch.cat([st, ct], dim=1)
                        logits = decoder.linear(concat_h)
                        prob_step = F.softmax(logits, dim=-1)[0]
                        prob_step = prob_step.cpu().numpy()
                        topn_index = prob_step.argsort()[:beam_width]
		        topn_index = [[j, i] for j in topn_index]
                        #print i, "topn_index",  topn_index 
	                #print "\n"
                        for j, beam_i in topn_index:
                            candidate = [seq + [j], score - np.log(prob_step[j]), beam_i]
                            all_candidates.append(candidate)
                    ordered = sorted(all_candidates, key=lambda x:x[1])
                    #print "ordered:%d"%len(ordered), ordered[:3]
                    candis = ordered[:beam_width]
		    print " ".join([str(e[-1]) for e in candis])
                res.append(candis[0][0])
                
    res_sents = id2word_np[np.array(res)]
    
    #unk->word
    assert len(val_data_unk)==len(res_sents)
    res_sents1=[]
    for unk2word, sent in zip(val_data_unk, res_sents):
        if unk2word=={}:
            res_sents1.append(sent)
        else:
            temp=[]
            for w in sent:
                try:
                    temp.append(unk2word[w])
                except:
                    temp.append(w)
            res_sents1.append(temp)
    res_sents=res_sents1
    return res_sents
#%%
if __name__ == '__main__':
    args = args_init()
    data_x, data_y = data_preparetion(args.input, args.output, limit_x=60, limit_y=50)
    all_text = read(args.all_text)
    
    word2id, id2word, word_vectors, embedding_weights, word_freq, nb_specials = Word2vec1.word2vec(all_text, use_bert=True)
    word2id, id2word, embedding_weights = unk_token(word2id, id2word, word_freq[:conf.vocab_size-nb_specials], word_vectors)
    id2word = invert_dict(word2id)

    data_x_id, word2unk = source2id(data_x, word2id)
    data_y_id = target2id(data_y, word2id, word2unk)
    unk2word = [invert_dict(e) for e in word2unk]
    
    k=0
    print " ".join([id2word[w] for w in data_x_id[k]])
    print " ".join([id2word[w] for w in data_y_id[k]])

    #%%
    #put END token to each target sequence
    data_y_id_=[]
    for e in data_y_id:
        data_y_id_.append(e+[word2id['END']])
    data_y_id = data_y_id_
    
    train_data_x_id, train_data_y_id, train_data_unk = data_x_id[1000:], data_y_id[1000:], unk2word[1000:]
    val_data_x_id,  val_data_y_id, val_data_unk = data_x_id[:1000], data_y_id[:1000], unk2word[:1000]
    
    bz = conf.batch_size
    nb = len(train_data_x_id)/bz
    #initialize src_encoder and decoder

    encoder = EncoderLSTM(embedding_weights, 
                         conf.enc_vocab_size,
                         conf.emb_dim, 
                         conf.hidden_dim,
                         conf.num_layers,
                         word2id,
                         id2word,
                         word2unk, 
                         word_vectors,
                         embedding_weights,
                         word_freq,
                         nb_specials)
    
    decoder = DecoderLSTM(embedding_weights, 
                         conf.enc_vocab_size,
                         conf.emb_dim, 
                         conf.hidden_dim*2,
                         conf.num_layers,
                         conf.batch_size).to(device)     
    
    '''
    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        # dim = 0 [30, xxx] -> [10, ...], [10, ...], [10, ...] on 3 GPUs
        src_encoder = nn.DataParallel(src_encoder)
        decoder = nn.DataParallel(decoder)
    '''
    encoder = encoder.to(device)      
    decoder = decoder.to(device)     
    encoder_h0 = encoder.initHidden(bz)
    encoder_optimizer = torch.optim.Adam(encoder.parameters(), lr=conf.lr)
    decoder_optimizer = torch.optim.Adam(decoder.parameters(), lr=conf.lr)
    
    id2word_np=[]
    for i in range(len(id2word)):
        id2word_np.append(id2word[i])
    id2word_np=np.array(id2word_np)
    
    for epoch in range(100):
        train_data = zip(train_data_x_id, train_data_y_id, train_data_unk)
        np.random.shuffle(train_data)
        train_data_x_id, train_data_y_id, train_data_unk = zip(*train_data)
        
        for b_i in range(nb):       
            train_scr_seqs, train_scr_mask = batch_padded(train_data_x_id[b_i*bz:(b_i+1)*bz])
            train_tgt_seqs, train_tgt_mask = batch_padded(train_data_y_id[b_i*bz:(b_i+1)*bz])
            encoder.zero_grad()
            decoder.zero_grad()
            
            hx_src, decoder_iniHidden = encoder(LongTensor(train_scr_seqs), train_scr_mask, encoder_h0)
            logits, target, length, pred = decoder(hx_src,
                                                   train_tgt_seqs,
                                                   train_scr_mask,
                                                   train_tgt_mask,
                                                   decoder_iniHidden)
           
            loss = compute_loss(logits, LongTensor(target), LongTensor(length))    
            if b_i%20==0:
                print "epoch:%d_nb:%d_loss:%f"%(epoch, b_i, loss) 
	    #print "epoch:%d_nb:%d_loss:%f"%(epoch, b_i, loss)                       
            loss.backward()
            encoder_optimizer.step()
            decoder_optimizer.step()
            loss = loss.item()
            if b_i%100==0:
                print "vocab:%d emb_dim:%d hdim:%d bz:%d lr:%f beam:%d"%(conf.vocab_size,conf.emb_dim,conf.hidden_dim,conf.batch_size,conf.lr,conf.beam)
                print "epoch:%d_nb:%d_loss:%f"%(epoch, b_i, loss)             
                res_sents=evaluate(bz, encoder, decoder, val_data_unk[:bz], val_data_x_id[:bz], val_data_y_id[:bz], id2word_np, conf.beam, max_length=MAX_LENGTH)              
                for i in range(min(40, len(res_sents))):
                    try:
                        k = res_sents[i].tolist().index('END')
                    except:
                        k = len(res_sents[i])
                    print "pred_sent%d:"%(i+1), " ".join(res_sents[i][:k])
                print "################################"
        torch.save(encoder, './save_model/encoder_beam%d_vocab_sz%d_bz%d'%(conf.beam, conf.vocab_size, conf.batch_size))
        torch.save(decoder, './save_model/decoder_beam%d_vocab_sz%d_bz%d'%(conf.beam, conf.vocab_size, conf.batch_size))
