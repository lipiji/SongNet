# coding=utf-8
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
import torch.multiprocessing as mp

from biglm import BIGLM
from data import Vocab, DataLoader
from adam import AdamWeightDecayOptimizer
from optim import Optim

import argparse, os
import random

def parse_config():
    parser = argparse.ArgumentParser()
    parser.add_argument('--embed_dim', type=int)
    parser.add_argument('--ff_embed_dim', type=int)
    parser.add_argument('--num_heads', type=int)
    parser.add_argument('--layers', type=int)
    parser.add_argument('--dropout', type=float)

    parser.add_argument('--train_data', type=str)
    parser.add_argument('--vocab', type=str)
    parser.add_argument('--min_occur_cnt', type=int)
    parser.add_argument('--batch_size', type=int)
    parser.add_argument('--warmup_steps', type=int)
    parser.add_argument('--lr', type=float)
    parser.add_argument('--smoothing', type=float)
    parser.add_argument('--weight_decay', type=float)
    parser.add_argument('--max_len', type=int)
    parser.add_argument('--min_len', type=int)
    parser.add_argument('--print_every', type=int)
    parser.add_argument('--save_every', type=int)
    parser.add_argument('--start_from', type=str, default=None)
    parser.add_argument('--save_dir', type=str)

    parser.add_argument('--approx', type=str, default='none')
    parser.add_argument('--fp16', action='store_true')
    parser.add_argument('--world_size', type=int)
    parser.add_argument('--gpus', type=int)
    parser.add_argument('--MASTER_ADDR', type=str)
    parser.add_argument('--MASTER_PORT', type=str)
    parser.add_argument('--start_rank', type=int)
    parser.add_argument('--backend', type=str)

    return parser.parse_args()

def update_lr(optimizer, lr):
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
 
def average_gradients(model):
    """ Gradient averaging. """
    normal = True
    size = float(dist.get_world_size())
    for param in model.parameters():
        if param.grad is not None:
            dist.all_reduce(param.grad.data, op=dist.ReduceOp.SUM)
            param.grad.data /= size
        else:
            normal = False
            break
    return normal

def run(args, local_rank):
    """ Distributed Synchronous """
    torch.manual_seed(1234)
    vocab = Vocab(args.vocab, min_occur_cnt=args.min_occur_cnt, specials=[])
    if (args.world_size == 1 or dist.get_rank() == 0):
        print (vocab.size, flush=True)
    model = BIGLM(local_rank, vocab, args.embed_dim, args.ff_embed_dim,\
                  args.num_heads, args.dropout, args.layers, args.smoothing, args.approx)
    if args.start_from is not None:
        ckpt = torch.load(args.start_from, map_location='cpu')
        model.load_state_dict(ckpt['model'])
    model = model.cuda(local_rank)
   
    weight_decay_params = []
    no_weight_decay_params = []
    
    for name, param in model.named_parameters():
        if name.endswith('bias') or 'layer_norm' in name:
            no_weight_decay_params.append(param)
        else:
            weight_decay_params.append(param)
    grouped_params = [{'params':weight_decay_params, 'weight_decay':args.weight_decay},
                        {'params':no_weight_decay_params, 'weight_decay':0.}]
    if args.world_size > 1:
        torch.manual_seed(1234 + dist.get_rank())
        random.seed(5678 + dist.get_rank())
    
    if args.fp16:
        try:
            from apex.optimizers import FP16_Optimizer
            from apex.optimizers import FusedAdam
        except ImportError:
            raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use fp16 training.")
        optimizer = FusedAdam(grouped_params,
                              lr=args.lr,
                              betas=(0.9, 0.999),
                              eps =1e-6,
                              bias_correction=False,
                              max_grad_norm=1.0)
        optimizer = FP16_Optimizer(optimizer, dynamic_loss_scale=True)

    else:
        if args.weight_decay > 0:
            optimizer = AdamWeightDecayOptimizer(grouped_params,
                           lr=args.lr, betas=(0.9, 0.999), eps=1e-6)
        else:
            optimizer = Optim(model.embed_dim, args.lr, args.warmup_steps, torch.optim.Adam(model.parameters(), lr=0, betas=(0.9, 0.998), eps=1e-9))

    if args.start_from is not None:
        optimizer.load_state_dict(ckpt['optimizer'])

    train_data = DataLoader(vocab, args.train_data, args.batch_size, args.max_len, args.min_len)
    batch_acm = 0
    acc_acm, nll_acm, ppl_acm, ntokens_acm, nxs, npairs_acm, loss_acm = 0., 0., 0., 0., 0., 0., 0.
    while True:
        model.train()
        if train_data.epoch_id > 30:
            break
        for xs_tpl, xs_seg, xs_pos, ys_truth, ys_inp, ys_tpl, ys_seg, ys_pos, msk in train_data:
            batch_acm += 1
            xs_tpl = xs_tpl.cuda(local_rank)
            xs_seg = xs_seg.cuda(local_rank)
            xs_pos = xs_pos.cuda(local_rank)
            ys_truth = ys_truth.cuda(local_rank)
            ys_inp = ys_inp.cuda(local_rank)
            ys_tpl = ys_tpl.cuda(local_rank)
            ys_seg = ys_seg.cuda(local_rank)
            ys_pos = ys_pos.cuda(local_rank)
            msk = msk.cuda(local_rank)

            model.zero_grad()
            res, loss, acc, nll, ppl, ntokens, npairs = model(xs_tpl, xs_seg, xs_pos, ys_truth, ys_inp, ys_tpl, ys_seg, ys_pos, msk)
            loss_acm += loss.item()
            acc_acm += acc
            nll_acm += nll
            ppl_acm += ppl
            ntokens_acm += ntokens
            npairs_acm += npairs
            nxs += npairs
            if args.fp16:
                optimizer.backward(loss)
            else:
                loss.backward()
            if args.world_size > 1:
                is_normal = average_gradients(model)
            else:
                is_normal = True
            if is_normal:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
            else:
                print("gradient: none, gpu: " + str(local_rank), flush=True)
                continue
            if (args.world_size==1 or dist.get_rank() ==0) and batch_acm%args.print_every == -1%args.print_every:
                print ('batch_acm %d, loss %.3f, acc %.3f, nll %.3f, ppl %.3f, x_acm %d, lr %.6f'\
                        %(batch_acm, loss_acm/args.print_every, acc_acm/ntokens_acm, \
                        nll_acm/nxs, ppl_acm/nxs, npairs_acm, optimizer._rate), flush=True)
                acc_acm, nll_acm, ppl_acm, ntokens_acm, loss_acm, nxs = 0., 0., 0., 0., 0., 0.
            if (args.world_size==1 or dist.get_rank() ==0) and batch_acm%args.save_every == -1%args.save_every:
                if not os.path.exists(args.save_dir):
                    os.mkdir(args.save_dir)
                torch.save({'args':args, 'model':model.state_dict(), 'optimizer':optimizer.state_dict()}, '%s/epoch%d_batch_%d'%(args.save_dir, train_data.epoch_id, batch_acm))

def init_processes(args, local_rank, fn, backend='nccl'):
    """ Initialize the distributed environment. """
    os.environ['MASTER_ADDR'] = args.MASTER_ADDR
    os.environ['MASTER_PORT'] = args.MASTER_PORT
    dist.init_process_group(backend, rank=args.start_rank + local_rank, world_size=args.world_size)
    fn(args, local_rank)

if __name__ == "__main__":
    mp.set_start_method('spawn')
    args = parse_config()

    if args.world_size == 1:
        run(args, 0)
        exit(0)
    processes = []
    for rank in range(args.gpus):
        p = mp.Process(target=init_processes, args=(args, rank, run, args.backend))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()
