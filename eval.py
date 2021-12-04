import argparse
import sys
import os

import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, utils
from vqvae import VQVAE
from tqdm import tqdm



def eval(model, epoch, dataloader, device='cuda', sample_size=8, datasetname='ffhq', partition='test'):
    criterion = nn.MSELoss()
    mse_sum = 0
    mse_n = 0
    for i, (img, label) in enumerate(loader):
        img = img.to(device)
        out, latent_loss = model(img)
        recon_loss = criterion(out, img)
        # latent_loss = latent_loss.mean()
        part_mse_sum = recon_loss.item() * img.shape[0]
        part_mse_n = img.shape[0] 
        mse_sum += part_mse_sum
        mse_n += part_mse_n
        perm = torch.randperm(img.shape[0])
        sampled_idx = perm[:sample_size]
        sampled_img, sampled_out = img[sampled_idx], out[sampled_idx]
    mse = mse_sum / mse_n
    utils.save_image(
        torch.cat([sampled_img, sampled_out], 0),
        f"out/{datasetname}/{partition}_{str(epoch).zfill(3)}_{mse}.png",
        nrow=sample_size,
        normalize=True,
        range=(-1, 1),
    )


if __name__ == '__main__':
    device = 'cuda'
    transform = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(256),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
        ]
    )
    test_set = datasets.ImageFolder('datasets/ffhq/test/', transform=transform)
    test_loader = tqdm(DataLoader(
        test_set, batch_size=32, num_workers=2
    ))
    train_set = datasets.ImageFolder('datasets/ffhq/train/', transform=transform)
    train_loader = tqdm(DataLoader(
        train_set, batch_size=32, num_workers=2
    ))
    loader_partiations = [[train_loader,'train'], [test_loader,'test']]
    with torch.no_grad():
        model = VQVAE().to(device)
        for i in range(0, 560, 50):
            ckpt = f'checkpoint/vqvae_{str(i+1).zfill(3)}.pt'
            model.load_state_dict(torch.load(ckpt))
            model.eval()
            for loader, partition in loader_partiations:
                eval(model, i+1, loader, partition=partition)

