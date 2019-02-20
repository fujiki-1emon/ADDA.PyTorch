from logging import getLogger
from time import time

import numpy as np
from sklearn.metrics import accuracy_score
import torch

from utils import AverageMeter, save


logger = getLogger('adda.trainer')


def train_source_cnn(
    source_cnn, train_loader, test_loader, criterion, optimizer,
    args=None
):
    best_score = None
    for epoch_i in range(1, 1 + args.epochs):
        start_time = time()
        training = train(
            source_cnn, train_loader, criterion, optimizer, args=args)
        validation = validate(
            source_cnn, test_loader, criterion, args=args)
        log = 'Epoch {}/{} '.format(epoch_i, args.epochs)
        log += 'Train/Loss {:.3f} Train/Acc {:.3f} '.format(
            training['loss'], training['acc'])
        log += 'Val/Loss {:.3f} Val/Acc {:.3f} '.format(
            validation['loss'], validation['acc'])
        log += 'Time {:.2f}s'.format(time() - start_time)
        logger.info(log)

        # save
        is_best = (best_score is None or validation['acc'] > best_score)
        best_score = validation['acc'] if is_best else best_score
        state_dict = {
            'model': source_cnn.state_dict(),
            'optimizer': optimizer.state_dict(),
            'epoch': epoch_i,
            'val/acc': best_score,
        }
        save(args.logdir, state_dict, is_best)

    return source_cnn


def train_target_cnn(
    source_cnn, target_cnn, discriminator,
    criterion, optimizer, d_optimizer,
    source_train_loader, target_train_loader, target_test_loader,
    args=None
):
    validation = validate(source_cnn, target_test_loader, criterion, args=args)
    log_source = 'Source/Val/Acc {:.3f} '.format(validation['acc'])

    # best_score = None
    for epoch_i in range(1, 1 + args.epochs):
        start_time = time()
        training = adversarial(
            source_cnn, target_cnn, discriminator,
            source_train_loader, target_train_loader,
            criterion, criterion,
            optimizer, d_optimizer,
            args=args
        )
        validation = validate(
            target_cnn, target_test_loader, criterion, args=args)
        log = 'Epoch {}/{} '.format(epoch_i, args.epochs)
        log += 'D/Loss {:.3f} Target/Loss {:.3f} '.format(
            training['d/loss'], training['target/loss'])
        log += 'Target/Val/Loss {:.3f} Target/Val/Acc {:.3f} '.format(
            validation['loss'], validation['acc'])
        log += log_source
        log += 'Time {:.2f}s'.format(time() - start_time)
        logger.info(log)


def adversarial(
    source_cnn, target_cnn, discriminator,
    source_loader, target_loader,
    criterion, d_criterion,
    optimizer, d_optimizer,
    args=None
):
    source_cnn.eval()
    target_cnn.train()
    discriminator.train()

    losses, d_losses = AverageMeter(), AverageMeter()
    n_iters = min(len(source_loader), len(target_loader))
    source_iter, target_iter = iter(source_loader), iter(target_loader)
    for iter_i in range(n_iters):
        source_data, source_target = source_iter.next()
        target_data, target_target = target_iter.next()
        source_data = source_data.to(args.device)
        target_data = target_data.to(args.device)
        bs = source_data.size(0)

        D_input_source = source_cnn.encoder(source_data)
        D_input_target = target_cnn.encoder(target_data)
        D_target_source = torch.tensor(
            [0] * bs, dtype=torch.long).to(args.device)
        D_target_target = torch.tensor(
            [1] * bs, dtype=torch.long).to(args.device)

        # train Discriminator
        D_output_source = discriminator(D_input_source)
        D_output_target = discriminator(D_input_target)
        d_loss_source = d_criterion(D_output_source, D_target_source)
        d_loss_target = d_criterion(D_output_target, D_target_target)
        d_loss = 0.5 * (d_loss_source + d_loss_target)
        d_optimizer.zero_grad()
        d_loss.backward(retain_graph=True)
        d_optimizer.step()
        d_losses.update(d_loss.item(), bs)

        # train Target
        '''
        D_input_target = target_cnn.encoder(target_data)
        D_output_target = discriminator(D_input_target)
        '''
        loss = criterion(D_output_target, D_target_source)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        losses.update(loss.item(), bs)
    return {'d/loss': d_losses.avg, 'target/loss': losses.avg}


def step(model, data, target, criterion, args):
    data, target = data.to(args.device), target.to(args.device)
    output = model(data)
    loss = criterion(output, target)
    return output, loss


def train(model, dataloader, criterion, optimizer, args=None):
    model.train()
    losses = AverageMeter()
    targets, probas = [], []
    for i, (data, target) in enumerate(dataloader):
        bs = target.size(0)
        output, loss = step(model, data, target, criterion, args)
        output = torch.softmax(output, dim=1)  # NOTE
        losses.update(loss.item(), bs)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        targets.extend(target.cpu().detach().numpy().tolist())
        probas.extend(output.cpu().detach().numpy().tolist())
    probas = np.asarray(probas)
    preds = np.argmax(probas, axis=1)
    acc = accuracy_score(targets, preds)
    return {
        'loss': losses.avg, 'acc': acc,
    }


def validate(model, dataloader, criterion, args=None):
    model.eval()
    losses = AverageMeter()
    targets, probas = [], []
    with torch.no_grad():
        for iter_i, (data, target) in enumerate(dataloader):
            bs = target.size(0)
            output, loss = step(model, data, target, criterion, args)
            output = torch.softmax(output, dim=1)  # NOTE: check
            losses.update(loss.item(), bs)
            targets.extend(target.cpu().numpy().tolist())
            probas.extend(output.cpu().numpy().tolist())
    probas = np.asarray(probas)
    preds = np.argmax(probas, axis=1)
    acc = accuracy_score(targets, preds)
    return {
        'loss': losses.avg, 'acc': acc,
    }
