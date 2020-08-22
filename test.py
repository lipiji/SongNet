import torch
from torch import nn
import torch.nn.functional as F
import random
import numpy as np
import copy 
import time

from biglm import BIGLM
from data import Vocab, DataLoader, s2t, s2xy



def init_seeds():
    random.seed(123)
    torch.manual_seed(123)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(123)

#init_seeds()

gpu = 1
def init_model(m_path, device, vocab):
    ckpt= torch.load(m_path, map_location='cpu')
    lm_args = ckpt['args']
    lm_vocab = Vocab(vocab, min_occur_cnt=lm_args.min_occur_cnt, specials=[])
    lm_model = BIGLM(device, lm_vocab, lm_args.embed_dim, lm_args.ff_embed_dim, lm_args.num_heads, lm_args.dropout, lm_args.layers, 0.1)
    lm_model.load_state_dict(ckpt['model'])
    lm_model = lm_model.cuda(device)
    lm_model.eval()
    return lm_model, lm_vocab, lm_args

m_path = "./ckpt/epoch7_batch_4999"
lm_model, lm_vocab, lm_args = init_model(m_path, gpu, "./data/vocab.txt")


k = 32
def top_k_inc(enc, src_padding_mask, inp_ys_tpl, inp_ys_seg, inp_ys_pos, s):
    start = time.time()
    incremental_state = None
    inp_y, m = s2t(s, lm_vocab)
    inp_y = inp_y.cuda(gpu)
    res = []
    for l in range(inp_ys_tpl.size(0)):
        probs, pred, incremental_state = lm_model.work_incremental(enc, src_padding_mask, \
                                         inp_y, inp_ys_tpl[0:l+1,:], inp_ys_seg[0:l+1,:], inp_ys_pos[0:l+1,:],\
                                         incremental_state)
        next_tk = []
        for i in range(len(s)):
            ctk = lm_vocab.idx2token(inp_ys_tpl[l,i].item())
            if ctk != "<c1>" and ctk != "<c2>" and ctk != "<c0>":
                next_tk.append(ctk)
                continue
            
            if l == 0:
                logits = probs[len(s[i]) - 1, i]
            else:
                logits = probs[0, i]
            ps, idx = torch.topk(logits, k=k)
            ps = ps / torch.sum(ps)
            sampled = torch.multinomial(ps, num_samples = 1)
            sampled_idx = idx[sampled]
            next_tk.append(lm_vocab.idx2token(sampled_idx.item()))
        
        s_ = []
        bidx = [1] * len(s)
        for idx, (sent, t) in enumerate(zip(s, next_tk)):
            if t == "<eos>":
                res.append(sent)
                bidx[idx] = 0
            else:
                s_.append(sent + [t])
        if not s_:
            break
        s = s_
        inp_y, m = s2t(s, lm_vocab)
        inp_y = inp_y.cuda(gpu)
        bidx = torch.BoolTensor(bidx).cuda(gpu)
        incremental_state["bidx"] = bidx
    res += s_
        
    #for i in res:
    #    print(''.join(i))
    print(time.time()-start)
    return res

def top_k(enc, src_padding_mask, inp_ys_tpl, inp_ys_seg, inp_ys_pos, s):
    inp_y, m = s2t(s, lm_vocab)
    inp_y = inp_y.cuda(gpu)

    start = time.time()
    res = []
    for l in range(inp_ys_tpl.size(0)):
        probs, pred = lm_model.work(enc, src_padding_mask, inp_y, inp_ys_tpl[0:l+1,:], inp_ys_seg[0:l+1,:], inp_ys_pos[0:l+1,:])
        next_tk = []
        for i in range(len(s)):
            ctk = lm_vocab.idx2token(inp_ys_tpl[l,i].item())
            if ctk != "<c1>":
                next_tk.append(ctk)
                continue
            logits = probs[len(s[i]) - 1, i]
            ps, idx = torch.topk(logits, k=k)
            ps = ps / torch.sum(ps)
            sampled = torch.multinomial(ps, num_samples = 1)
            sampled_idx = idx[sampled]
            next_tk.append(lm_vocab.idx2token(sampled_idx.item()))
        
        s_ = []
        for sent, t in zip(s, next_tk):
            if t == "<eos>":
                res.append(sent)
            else:
                s_.append(sent + [t])
        if not s_:
            break
        s = s_
        inp_y, m = s2t(s, lm_vocab)
        inp_y = inp_y.cuda(gpu)

    res += s_
        
    #for i in res:
    #    print(''.join(i))

    #print(time.time()-start)
    return res
 
    
def greedy(enc, src_padding_mask, inp_ys_tpl, inp_ys_seg, inp_ys_pos, s):
    start = time.time()
    incremental_state = None
    inp_y, m = s2t(s, lm_vocab)
    inp_y = inp_y.cuda(gpu)
    res = []
    for l in range(inp_ys_tpl.size(0)):
        probs, pred, incremental_state = lm_model.work_incremental(enc, src_padding_mask, \
                                         inp_y, inp_ys_tpl[0:l+1,:], inp_ys_seg[0:l+1,:], inp_ys_pos[0:l+1,:],\
                                         incremental_state)
        next_tk = []
        for i in range(len(s)):
            ctk = lm_vocab.idx2token(inp_ys_tpl[l,i].item())
            if ctk != "<c1>" and ctk != "<c2>" and ctk != "<c0>":
                next_tk.append(ctk)
                continue
            
            if l == 0:
                pred = pred[len(s[i]) - 1, i]
            else:
                pred = pred[0, i]
            next_tk.append(lm_vocab.idx2token(pred.item()))
        
        s_ = []
        bidx = [1] * len(s)
        for idx, (sent, t) in enumerate(zip(s, next_tk)):
            if t == "<eos>":
                res.append(sent)
                bidx[idx] = 0
            else:
                s_.append(sent + [t])
        if not s_:
            break
        s = s_
        inp_y, m = s2t(s, lm_vocab)
        inp_y = inp_y.cuda(gpu)
        bidx = torch.BoolTensor(bidx).cuda(gpu)
        incremental_state["bidx"] = bidx
    res += s_
        
    #for i in res:
    #    print(''.join(i))
    print(time.time()-start)
    return res


def beam_decode(s, x, enc, src_padding_mask, inp_ys_tpl, inp_ys_seg, inp_ys_pos):
    beam_size = 5
    
    num_live = 1
    num_dead = 0 
    samples = []
    sample_scores = np.zeros(beam_size)

    last_traces = [[]]
    last_scores = torch.FloatTensor(np.zeros(1)).to(gpu)

    x = x.to(gpu)
    ys = x

    for l in range(inp_ys_tpl.size(0)):
        seq_len, bsz = ys.size()
        enc_ = enc.repeat(1, bsz, 1)
        src_padding_mask_ = src_padding_mask.repeat(1, bsz)
        inp_ys_tpl_ = inp_ys_tpl.repeat(1, bsz)
        inp_ys_seg_ = inp_ys_seg.repeat(1, bsz)
        inp_ys_pos_ = inp_ys_pos.repeat(1, bsz)

        y_pred, _ = lm_model.work(enc_, src_padding_mask_, ys, inp_ys_tpl_[0:l+1,:], inp_ys_seg_[0:l+1,:], inp_ys_pos_[0:l+1,:])

        dict_size = y_pred.shape[-1]
        y_pred = y_pred[-1, :, :] 

        cand_y_scores = last_scores + torch.log(y_pred) # larger is better
        cand_scores = cand_y_scores.flatten()
        idx_top_joint_scores = torch.topk(cand_scores, beam_size - num_dead)[1]
        
        '''
        ps, idx_top_joint_scores = torch.topk(cand_scores, 100)
        ps = F.softmax(ps)
        sampled = torch.multinomial(ps, num_samples = beam_size - num_dead)
        idx_top_joint_scores = idx_top_joint_scores[sampled]
        '''

        idx_last_traces = idx_top_joint_scores / dict_size
        idx_word_now = idx_top_joint_scores % dict_size
        top_joint_scores = cand_scores[idx_top_joint_scores]

        traces_now = []
        scores_now = np.zeros((beam_size - num_dead))
        ys_now = []
        for i, [j, k] in enumerate(zip(idx_last_traces, idx_word_now)):
            traces_now.append(last_traces[j] + [k])
            scores_now[i] = copy.copy(top_joint_scores[i])
            ys_now.append(copy.copy(ys[:,j]))


        num_live = 0  
        last_traces = []
        last_scores = []
        ys = []
        for i in range(len(traces_now)):
            w = lm_vocab.idx2token(traces_now[i][-1].item())
            if w == "<eos>":
                samples.append([str(e.item()) for e in traces_now[i][:-1]])
                sample_scores[num_dead] = scores_now[i] 
                num_dead += 1
            else:
                last_traces.append(traces_now[i])
                last_scores.append(scores_now[i])
                ys.append(ys_now[i])
                num_live += 1
        
        if num_live == 0 or num_dead >= beam_size:
            break
        ys = torch.stack(ys, dim = 1) 

        last_scores = torch.FloatTensor(np.array(last_scores).reshape((num_live, 1))).to(gpu)
        next_y = []
        for e in last_traces:
            eid = e[-1].item()
            next_y.append(eid)
        next_y = np.array(next_y).reshape((1, num_live))
        next_y = torch.LongTensor(next_y).to(gpu)
        
        ys = torch.cat([ys, next_y], dim=0)
       
        assert num_live + num_dead == beam_size 
        # end for loop

    if num_live > 0:
        for i in range(num_live):
            samples.append([str(e.item()) for e in last_traces[i]])
            sample_scores[num_dead] = last_scores[i]
            num_dead += 1  

    idx_sorted_scores = np.argsort(sample_scores) # ascending order

    sorted_samples = []
    sorted_scores = []
    filter_idx = []
    for e in idx_sorted_scores:
        if len(samples[e]) > 0:
            filter_idx.append(e)
    if len(filter_idx) == 0:
        filter_idx = idx_sorted_scores
    for e in filter_idx:
        sorted_samples.append(samples[e])
        sorted_scores.append(sample_scores[e])

    res = []
    dec_words = []
    for sample in sorted_samples[::-1]:
        for e in sample:
            e = int(e)
            dec_words.append(lm_vocab.idx2token(e))
        #r = ''.join(dec_words)
        #print(r)
        res.append(dec_words)
        dec_words = []

    return res


def beam_search(enc, src_padding_mask, ys_tpl, ys_seg, ys_pos, s):
    x, m = s2t(s, lm_vocab)
    return beam_decode(s[0], x, enc, src_padding_mask, ys_tpl, ys_seg, ys_pos)


ds = []
with open("./data/test.txt", "r") as f:
    for line in f:
        line = line.strip()
        if line:
            ds.append(line)
print(len(ds))

local_rank = gpu
batch_size = 1
cp_size = 1
batches = round(len(ds) / batch_size)

for i in range(5): 
    idx = 0
    fo = open("./results/top-"+str(k)+"/out"+str(i+1)+".txt", "w")
    while idx < len(ds):
        lb = ds[idx:idx + batch_size]
        cplb = []
        for line in lb:
            cplb += [line for i in range(cp_size)]
        print(cplb) 
        xs_tpl, xs_seg, xs_pos, \
        ys_truth, ys_inp, \
        ys_tpl, ys_seg, ys_pos, msk = s2xy(cplb, lm_vocab, lm_args.max_len, 2)

        xs_tpl = xs_tpl.cuda(local_rank)
        xs_seg = xs_seg.cuda(local_rank)
        xs_pos = xs_pos.cuda(local_rank)
        ys_tpl = ys_tpl.cuda(local_rank)
        ys_seg = ys_seg.cuda(local_rank)
        ys_pos = ys_pos.cuda(local_rank)

        enc, src_padding_mask = lm_model.encode(xs_tpl, xs_seg, xs_pos)
        s = [['<bos>']] * batch_size * cp_size   
        res = top_k_inc(enc, src_padding_mask, ys_tpl, ys_seg, ys_pos, s)

        for i, line in enumerate(cplb):
            r = ''.join(res[i])
            print(line)
            print(r)
    
            fo.write(line + "\t" + r + "\n")
    
        idx += batch_size
    
    fo.close()
