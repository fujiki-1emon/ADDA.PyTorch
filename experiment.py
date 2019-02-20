import os

import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision.datasets import SVHN, MNIST
from torchvision import transforms

from models import CNN, Discriminator
from trainer import train_target_cnn
from utils import get_logger


def run(args):
    if not os.path.exists(args.logdir):
        os.makedirs(args.logdir)
    logger = get_logger(os.path.join(args.logdir, 'main.log'))
    logger.info(args)

    # data
    source_transform = transforms.Compose([
        # transforms.Grayscale(),
        transforms.ToTensor()]
    )
    target_transform = transforms.Compose([
        transforms.Resize(32),
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x.repeat(3, 1, 1))
    ])
    source_dataset_train = SVHN(
        './input', 'train', transform=source_transform, download=True)
    target_dataset_train = MNIST(
        './input', 'train', transform=target_transform, download=True)
    target_dataset_test = MNIST(
        './input', 'test', transform=target_transform, download=True)
    source_train_loader = DataLoader(
        source_dataset_train, args.batch_size, shuffle=True,
        drop_last=True,
        num_workers=args.n_workers)
    target_train_loader = DataLoader(
        target_dataset_train, args.batch_size, shuffle=True,
        drop_last=True,
        num_workers=args.n_workers)
    target_test_loader = DataLoader(
        target_dataset_test, args.batch_size, shuffle=False,
        num_workers=args.n_workers)

    # train source CNN
    source_cnn = CNN(in_channels=args.in_channels).to(args.device)
    if os.path.isfile(args.trained):
        c = torch.load(args.trained)
        source_cnn.load_state_dict(c['model'])
        logger.info('Loaded `{}`'.format(args.trained))

    # train target CNN
    target_cnn = CNN(in_channels=args.in_channels, target=True).to(args.device)
    target_cnn.load_state_dict(source_cnn.state_dict())
    discriminator = Discriminator(args=args).to(args.device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(
        target_cnn.encoder.parameters(),
        lr=args.lr, betas=args.betas, weight_decay=args.weight_decay)
    d_optimizer = optim.Adam(
        discriminator.parameters(),
        lr=args.lr, betas=args.betas, weight_decay=args.weight_decay)
    train_target_cnn(
        source_cnn, target_cnn, discriminator,
        criterion, optimizer, d_optimizer,
        source_train_loader, target_train_loader, target_test_loader,
        args=args)
