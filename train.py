from tqdm import tqdm
import numpy as np
from PIL import Image
from math import log, sqrt, pi
import os

import argparse

import torch
from torch import nn, optim
from torch.autograd import Variable, grad
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, utils

from datasets import Dots
from model import Glow
from discriminator import Discriminator
import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

parser = argparse.ArgumentParser(description="Glow trainer")

parser.add_argument("--dataset", default='dots', type=str, help="dataset name")
parser.add_argument(
    "--noisy", action="store_true", help="noisy"
)
parser.add_argument("--n_dots", default=3, type=int, help="dataset selection for dots")

parser.add_argument("--batch", default=16, type=int, help="batch size")
parser.add_argument("--iter", default=200000, type=int, help="maximum iterations")
parser.add_argument(
    "--n_flow", default=32, type=int, help="number of flows in each block"
)
parser.add_argument("--n_block", default=4, type=int, help="number of blocks")
parser.add_argument(
    "--no_lu",
    action="store_true",
    help="use plain convolution instead of LU decomposed version",
)
parser.add_argument(
    "--affine", action="store_true", help="use affine coupling instead of additive"
)
parser.add_argument("--n_bits", default=5, type=int, help="number of bits")
parser.add_argument("--lr", default=1e-4, type=float, help="learning rate")
parser.add_argument("--img_size", default=64, type=int, help="image size")
parser.add_argument("--temp", default=0.7, type=float, help="temperature of sampling")
parser.add_argument("--n_sample", default=20, type=int, help="number of samples")
parser.add_argument("path", metavar="PATH", type=str, help="Path to image directory")

parser.add_argument("--workdir", default='results/exp_test', type=str, help="workdir name")
parser.add_argument("--logfile", default='log', type=str, help="logfile name")


def sample_data(path, batch_size, image_size):
    transform = transforms.Compose(
        [
            transforms.Resize(image_size),
            transforms.CenterCrop(image_size),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
        ]
    )

    dataset = datasets.ImageFolder(path, transform=transform)
    loader = DataLoader(dataset, shuffle=True, batch_size=batch_size, num_workers=4)
    loader = iter(loader)

    while True:
        try:
            yield next(loader)

        except StopIteration:
            loader = DataLoader(
                dataset, shuffle=True, batch_size=batch_size, num_workers=4
            )
            loader = iter(loader)
            yield next(loader)


def sample_data_dots(path, batch_size, image_size, args):
    imgs = []
    labels = []
    db_path = [os.path.join('data/dots/', f'{args.n_dots}_dots')]
    db_files = [os.listdir(path) for path in db_path]
    for db_file in db_files[0]:
        filename = os.path.join(db_path[0], db_file)
        img = np.load(filename)['images']
        imgs.append(img)
        labels.append(3*np.ones(shape=img.shape[0]))

    train_imgs = np.concatenate(imgs[:-1])
    train_imgs = torch.Tensor(train_imgs).permute(0, 3, 1, 2)
    train_labels = np.concatenate(labels[:-1])
    train_labels = torch.Tensor(train_labels)

    dataset = Dots(train_imgs, train_labels, noisy=args.noisy, img_size=image_size)
    loader = DataLoader(dataset, shuffle=True, batch_size=batch_size, num_workers=4)
    loader = iter(loader)

    while True:
        try:
            yield next(loader)

        except StopIteration:
            loader = DataLoader(
                dataset, shuffle=True, batch_size=batch_size, num_workers=4
            )
            loader = iter(loader)
            yield next(loader)


def calc_z_shapes(n_channel, input_size, n_flow, n_block):
    z_shapes = []

    for i in range(n_block - 1):
        input_size //= 2
        n_channel *= 2

        z_shapes.append((n_channel, input_size, input_size))

    input_size //= 2
    z_shapes.append((n_channel * 4, input_size, input_size))

    return z_shapes


def calc_loss(log_p, logdet, image_size, n_bins):
    # log_p = calc_log_p([z_list])
    n_pixel = image_size * image_size * 3

    loss = -log(n_bins) * n_pixel
    loss = loss + logdet + log_p

    return (
        (-loss / (log(2) * n_pixel)).mean(),
        (log_p / (log(2) * n_pixel)).mean(),
        (logdet / (log(2) * n_pixel)).mean(),
    )


def permute_dims(z):
    assert z.dim() == 2

    B, _ = z.size()
    perm_z = []
    for z_j in z.split(1, 1):
        perm = torch.randperm(B).to(z.device)
        perm_z_j = z_j[perm]
        perm_z.append(perm_z_j)

    return torch.cat(perm_z, 1)


def train(args, model, optimizer, discriminator=None, optimizer_disc=None):

    # log
    os.makedirs(f'{args.workdir}/logs', exist_ok=True)
    os.makedirs(f'{args.workdir}/checkpoint', exist_ok=True)
    os.makedirs(f'{args.workdir}/sample', exist_ok=True)
    log = open(f'{args.workdir}/logs/{args.logfile}.txt', 'a')
    log_args = '========== Options ==========\n'
    args_var = vars(args)
    for k, v in args_var.items():
        log_args += f'{str(k)}: {str(v)}\n'
    log_args += '=============================\n'
    log.write(log_args)
    log.close()

    if args.dataset == 'dots':
        dataset = iter(sample_data_dots(args.path, args.batch, args.img_size, args))
    else:
        dataset = iter(sample_data(args.path, args.batch, args.img_size))
    n_bins = 2.0 ** args.n_bits

    z_sample = []
    z_shapes = calc_z_shapes(3, args.img_size, args.n_flow, args.n_block)
    for z in z_shapes:
        z_new = torch.randn(args.n_sample, *z) * args.temp
        z_sample.append(z_new.to(device))

    ones = torch.ones(args.batch, dtype=torch.long, device=device)
    zeros = torch.zeros(args.batch, dtype=torch.long, device=device)

    with tqdm(range(args.iter)) as pbar:
        for i in pbar:
            image, _ = next(dataset)
            image = image.to(device)

            image = image * 255

            if args.n_bits < 8:
                image = torch.floor(image / 2 ** (8 - args.n_bits))

            image = image / n_bins - 0.5

            if i == 0:
                with torch.no_grad():
                    log_p, logdet, _ = model.module(
                        image + torch.rand_like(image) / n_bins
                    )

                torch.save(
                    model.state_dict(), f"{args.workdir}/checkpoint/model_{str(i + 1).zfill(6)}.pt"
                )
                torch.save(
                    optimizer.state_dict(), f"{args.workdir}/checkpoint/optim_{str(i + 1).zfill(6)}.pt"
                )
                torch.save(
                    discriminator.state_dict(), f"{args.workdir}/checkpoint/model_disc_{str(i + 1).zfill(6)}.pt"
                )
                torch.save(
                    optimizer_disc.state_dict(), f"{args.workdir}/checkpoint/optim_disc_{str(i + 1).zfill(6)}.pt"
                )
                continue

            else:
                log_p, logdet, z_outs = model(image + torch.rand_like(image) / n_bins)

            logdet = logdet.mean()

            ####
            #z_concat = model.module.z_outs_concat(z_outs)
            z_concat = z_outs.view(z_outs.size(0), -1)
            d_z = discriminator(z_concat)
            loss_tc = (d_z[:, :1] - d_z[:, 1:]).mean()
            ####

            loss, log_p, log_det = calc_loss(log_p, logdet, args.img_size, n_bins)
            loss = loss + loss_tc / (np.log(2) * args.img_size * args.img_size * 3)
            model.zero_grad()
            loss.backward(retain_graph=True)
            # warmup_lr = args.lr * min(1, i * batch_size / (50000 * 10))
            warmup_lr = args.lr
            optimizer.param_groups[0]["lr"] = warmup_lr


            ####
            image2, _ = next(dataset)
            image2 = image2.to(device)

            image2 = image2 * 255

            if args.n_bits < 8:
                image2 = torch.floor(image2 / 2 ** (8 - args.n_bits))

            image2 = image2 / n_bins - 0.5
            z_prime = model(image2 + torch.rand_like(image2) / n_bins, need_det=False)
            #z_prime = model.module.z_outs_concat(z_outs2)
            z_prime = z_prime.view(z_prime.size(0), -1)
            z_pperm = permute_dims(z_prime).detach()
            d_z_pperm = discriminator(z_pperm)

            loss_tc_disc = 0.5 * (F.cross_entropy(d_z, zeros) + F.cross_entropy(d_z_pperm, ones))

            discriminator.zero_grad()
            loss_tc_disc.backward()

            optimizer.step()
            optimizer_disc.step()
            ####

            # pbar.set_description(
            #     f"Loss: {loss.item():.5f}; logP: {log_p.item():.5f}; logdet: {log_det.item():.5f}; lr: {warmup_lr:.7f}"
            # )
            pbar.set_description(
                f"Loss: {loss.item():.3f}; logP: {log_p.item():.3f}; logdet: {log_det.item():.3f}; loss_tc: {loss_tc.item():.3f}; loss_tc_disc: {loss_tc_disc.item():.3f}; lr: {warmup_lr:.5f}"
            )

            log = open(f'{args.workdir}/logs/{args.logfile}.txt', 'a')
            # log.write(f'Iter: {i+1:6d}; Loss: {loss.item():.5f}; logP: {log_p.item():.5f}; logdet: {log_det.item():.5f}; lr: {warmup_lr:.7f}\n')
            log.write(f'Iter: {i+1:6d}; Loss: {loss.item():.5f}; logP: {log_p.item():.5f}; logdet: {log_det.item():.5f}; loss_tc: {loss_tc.item():.5f}; loss_tc_disc: {loss_tc_disc.item():.5f}; lr: {warmup_lr:.7f}\n')
            log.close()

            if i % 1000 == 0:
                with torch.no_grad():
                    utils.save_image(
                        model_single.reverse(z_sample).cpu().data,
                        f"{args.workdir}/sample/{str(i + 1).zfill(6)}.png",
                        normalize=True,
                        nrow=10,
                        range=(-0.5, 0.5),
                    )

            if i % 10000 == 0:
                torch.save(
                    model.state_dict(), f"{args.workdir}/checkpoint/model_{str(i + 1).zfill(6)}.pt"
                )
                torch.save(
                    optimizer.state_dict(), f"{args.workdir}/checkpoint/optim_{str(i + 1).zfill(6)}.pt"
                )
                torch.save(
                    discriminator.state_dict(), f"{args.workdir}/checkpoint/model_disc_{str(i + 1).zfill(6)}.pt"
                )
                torch.save(
                    optimizer_disc.state_dict(), f"{args.workdir}/checkpoint/optim_disc_{str(i + 1).zfill(6)}.pt"
                )


if __name__ == "__main__":
    args = parser.parse_args()
    print(args)

    model_single = Glow(
        3, args.n_flow, args.n_block, affine=args.affine, conv_lu=not args.no_lu
    )
    model = nn.DataParallel(model_single)
    # model = model_single
    model = model.to(device)

    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    discriminator = Discriminator(3 * args.img_size * args.img_size)
    discriminator = nn.DataParallel(discriminator)
    discriminator = discriminator.to(device)
    optimizer_disc = optim.Adam(discriminator.parameters(), lr=args.lr, betas=(0.5, 0.9))

    train(args, model, optimizer, discriminator, optimizer_disc)
