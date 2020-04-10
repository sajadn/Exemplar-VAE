from __future__ import print_function
import torch
import torch.utils.data
import math

min_epsilon = 1e-5
max_epsilon = 1.-1e-5
log_sigmoid = torch.nn.LogSigmoid()
log_2_pi = math.log(2*math.pi)


def pairwise_distance(z, means):
    z = z.double()
    means = means.double()
    dist1 = (z**2).sum(dim=1).unsqueeze(1).expand(-1, means.shape[0]) #MB x C
    dist2 = (means**2).sum(dim=1).unsqueeze(0).expand(z.shape[0], -1) #MB x C
    dist3 = torch.mm(z, torch.transpose(means, 0, 1)) #MB x C
    return (dist1 + dist2 + - 2*dist3).float()


def log_normal_diag_vectorized(x, mean, log_var):
    log_var_sqrt = log_var.mul(0.5).exp_()
    pair_dist = pairwise_distance(x/log_var_sqrt, mean/log_var_sqrt)
    log_normal = -0.5 * torch.sum(log_var+log_2_pi, dim=1) - 0.5*pair_dist
    return log_normal, pair_dist


def log_normal_diag(x, mean, log_var, average=False, dim=None):
    log_normal = -0.5 * (log_var + log_2_pi + torch.pow( x - mean, 2 ) / torch.exp( log_var ) )
    if average:
        return torch.mean(log_normal, dim)
    else:
        return torch.sum(log_normal, dim)


def log_normal_standard(x, average=False, dim=None):
    log_normal = -0.5 * torch.pow(x, 2) - 0.5 * log_2_pi*x.new_ones(size=x.shape)
    if average:
        return torch.mean(log_normal, dim)
    else:
        return torch.sum(log_normal, dim)


def log_bernoulli(x, mean, average=False, dim=None):
    probs = torch.clamp( mean, min=min_epsilon, max=max_epsilon)
    log_bernoulli = x * torch.log(probs) + (1. - x) * torch.log(1. - probs)

    if average:
        return torch.mean(log_bernoulli, dim)
    else:
        return torch.sum(log_bernoulli, dim)


def log_logistic_256(x, mean, logvar, average=False, reduce=True, dim=None):
    bin_size = 1. / 256.
    # implementation like https://github.com/openai/iaf/blob/master/tf_utils/distributions.py#L28
    scale = torch.exp(logvar)
    x = (torch.floor(x / bin_size) * bin_size - mean) / scale
    cdf_plus = torch.sigmoid(x + bin_size/scale)
    cdf_minus = torch.sigmoid(x)
    log_logist_256 = torch.log(cdf_plus - cdf_minus + 1e-7)

    if average:
        return torch.mean(log_logist_256, dim)
    else:
        return torch.sum(log_logist_256, dim)

