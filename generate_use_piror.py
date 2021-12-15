import argparse
from threading import Condition

import numpy as np
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from tqdm import tqdm


from dataset import LMDBDataset
from pixelsnail import PixelSNAIL
from scheduler import CycleScheduler
from vqvae import VQVAE
from torchvision import utils
from tqdm import tqdm
from torch.distributions import *

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--numsample', type=int, default=6)
    parser.add_argument('--ckpt_top', type=str, default='checkpoint/pixelsnail_top_420.pt')
    parser.add_argument('--ckpt_bottom', type=str, default='checkpoint/pixelsnail_bottom_227.pt')
    parser.add_argument('--ckpt_vqvae', type=str, default='checkpoint/vqvae_560.pt')
    parser.add_argument('--hier', type=str, default='top')
    parser.add_argument('--outpath', type=str, default='out/')
    parser.add_argument('--testset', type=str, default='ffhqtest')
    args = parser.parse_args()

    num_sample = args.numsample
    device = 'cuda'
    with torch.no_grad():
        dataset = LMDBDataset(args.testset)
        loader = DataLoader(
        dataset, batch_size=args.numsample, shuffle=True, num_workers=4, drop_last=True
        )
        top, bottom, label = next(iter(loader))
        # random sample
        # top, bottom = torch.randint(512, top.shape), torch.randint(512, bottom.shape)
        top, bottom = torch.zeros(top.shape).long().cuda(), torch.zeros(bottom.shape).long().cuda()
        # top_ids
        ckpt_top = torch.load(args.ckpt_top)
        top_args = ckpt_top['args']
        top_model = PixelSNAIL(
            [32, 32],
            512,
            top_args.channel,
            5,
            4,
            top_args.n_res_block,
            top_args.n_res_channel,
            dropout=top_args.dropout,
            n_out_res_block=top_args.n_out_res_block,
        )
        top_model.load_state_dict(ckpt_top['model'])
        top_model = top_model.to(device)
        top_model.eval()
        # samples_id_t = torch.randint(512, [num_sample,32,32])
        samples_id_t = top
        for row in tqdm(range(samples_id_t.shape[1])):
            for col in range(samples_id_t.shape[2]):
                id_t_logits = top_model(samples_id_t)[0].permute(0, 2, 3, 1)
                m = Categorical(logits=id_t_logits)
                id_t = m.sample()
                # _, id_t = id_t_logits.max(1)
                samples_id_t[:, row, col] = id_t[:, row, col]
        id_t = samples_id_t
        # id_t_logits = top_model(samples_id_t)[0]
        # _, id_t = id_t_logits.max(1)
        # bottom_ids
        ckpt_bottom = torch.load(args.ckpt_bottom) 
        bottom_args = ckpt_bottom['args']
        bottom_model = PixelSNAIL(
        [64, 64],
        512,
        bottom_args.channel,
        5,
        4,
        bottom_args.n_res_block,
        bottom_args.n_res_channel,
        attention=False,
        dropout=bottom_args.dropout,
        n_cond_res_block=bottom_args.n_cond_res_block,
        cond_res_channel=bottom_args.n_res_channel,
        )
        bottom_model.load_state_dict(ckpt_bottom['model'])
        bottom_model = bottom_model.to(device)
        # samples_id_b = torch.randint(512, (num_sample, 64, 64))
        samples_id_b = bottom
        # id_b_logits = bottom_model(samples_id_b, condition=id_t)[0]
        # _, id_b = id_b_logits.max(1)
        for row in tqdm(range(samples_id_b.shape[1])):
            for col in range(samples_id_b.shape[2]):
                id_b_logits = bottom_model(samples_id_b, condition=id_t)[0].permute(0, 2, 3, 1)
                m = Categorical(logits=id_b_logits)
                id_b = m.sample()
                # _, id_b = id_b_logits.max(1)
                samples_id_b[:, row, col] = id_b[:, row, col]
        id_b = samples_id_b
        # _, id_b = id_b_logits.max(1)
        # Decoding
        vqvae = VQVAE()
        vqvae.load_state_dict(torch.load(args.ckpt_vqvae))
        vqvae = vqvae.to(device)
        vqvae = vqvae.eval()
        res_t_b = vqvae.decode_code(id_t, id_b)
        res_t = vqvae.decode_code(id_t, id_b*0)
        res_b = vqvae.decode_code(id_t*0, id_b)
        utils.save_image(
            torch.cat([res_t, res_b, res_t_b], 0),
            f"out/{num_sample}_reencode_prior.png",
            nrow=num_sample,
            normalize=True,
            range=(-1, 1),
        )
        # reconstruction
        top = top.to(device)
        bottom = bottom.to(device)
        res_t_b = vqvae.decode_code(top, bottom)
        res_t = vqvae.decode_code(top, bottom*0)
        res_b = vqvae.decode_code(top*0, bottom) 
        utils.save_image(
            torch.cat([res_t, res_b, res_t_b], 0),
            f"out/{num_sample}_reconstruction.png",
            nrow=num_sample,
            normalize=True,
            range=(-1, 1),
        )
        print('done')
