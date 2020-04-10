from __future__ import print_function
import torch


def set_beta(args, epoch):
    if args.warmup == 0:
        beta = 1.
    else:
        beta = 1. * epoch / args.warmup
        if beta > 1.:
            beta = 1.
    return beta


def train_one_epoch(epoch, args, train_loader, model, optimizer):
    train_loss, train_re, train_kl = 0, 0, 0
    model.train()
    beta = set_beta(args, epoch)
    print('beta: {}'.format(beta))
    if args.approximate_prior is True:
        with torch.no_grad():
            cached_z, cached_log_var = model.cache_z(train_loader.dataset)
            cache = (cached_z, cached_log_var)
    else:
        cache = None

    for batch_idx, (data, indices, target) in enumerate(train_loader):
        data, indices, target = data.to(args.device), indices.to(args.device), target.to(args.device)

        if args.dynamic_binarization:
            x = torch.bernoulli(data)
        else:
            x = data

        x = (x, indices)
        optimizer.zero_grad()
        loss, RE, KL = model.calculate_loss(x, beta, average=True, cache=cache, dataset=train_loader.dataset)
        loss.backward()
        optimizer.step()

        with torch.no_grad():
            train_loss += loss.data.item()
            train_re += -RE.data.item()
            train_kl += KL.data.item()
            if cache is not None:
                cache = (cache[0].detach(), cache[1].detach())

    train_loss /= len(train_loader)
    train_re /= len(train_loader)
    train_kl /= len(train_loader)
    return train_loss, train_re, train_kl
