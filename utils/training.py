from __future__ import print_function
import torch
from utils.plot_images import imshow
import matplotlib.pylab as plt
from utils.utils import scaled_logit, inverse_scaled_logit, scaled_logit_torch

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

    total = 0
    for batch_idx, (data, indices, target) in enumerate(train_loader):
        data, indices, target = data.to(args.device).squeeze(), indices.to(args.device), target.to(args.device)
        if args.dynamic_binarization:
            x = torch.bernoulli(data)
        elif args.use_logit:
            x = inverse_scaled_logit(data, args.lambd) + (data.new_empty(size=data.shape).uniform_() -0.5)/256

            x = scaled_logit_torch(x, args.lambd)
        else:
            x = data

        x = (x, indices)
        optimizer.zero_grad()
        loss, RE, KL = model.calculate_loss(x, beta, average=True, cache=cache, dataset=train_loader.dataset)
        loss.backward()
        optimizer.step()

        total += len(data)
        with torch.no_grad():
            train_loss += loss.data.item()*len(data)
            train_re += -RE.data.item()*len(data)
            train_kl += KL.data.item()*len(data)
            if cache is not None:
                cache = (cache[0].detach(), cache[1].detach())
    train_loss /= total
    train_re /= total
    train_kl /= total
    return train_loss, train_re, train_kl
