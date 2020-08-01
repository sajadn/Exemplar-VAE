from __future__ import print_function
import torch
import torch.utils.data
import math
import torch.nn.functional as F
import numpy as np

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

#def log_logistic_256(x, mean, logvar, average=False, reduce=True, dim=None):
#    bin_size = 1. / 256.

    # implementation like https://github.com/openai/iaf/blob/master/tf_utils/distributions.py#L28
#    scale = torch.exp(logvar)
#    x = (torch.floor(x / bin_size) * bin_size - mean) / scale
#    cdf_plus = torch.sigmoid(x + bin_size/scale)
#    cdf_minus = torch.sigmoid(x)

    # calculate final log-likelihood for an image
#    log_logist_256 =  torch.log(cdf_plus - cdf_minus + 1.e-7)

#    if average:
#        return torch.mean(log_logist_256, dim)
#    else:
#        return torch.sum(log_logist_256, dim)

def to_one_hot(tensor, n, fill_with=1.):
    # we perform one hot encore with respect to the last axis
    one_hot = torch.FloatTensor(tensor.size() + (n,)).zero_()
    if tensor.is_cuda: one_hot = one_hot.cuda()
    one_hot.scatter_(len(tensor.size()), tensor.unsqueeze(-1), fill_with)
    return one_hot.detach()

def sample_from_discretized_mix_logistic(l, nr_mix):
    # Pytorch ordering
    l = l.permute(0, 2, 3, 1)
    ls = [int(y) for y in l.size()]
    xs = ls[:-1] + [3]

    # unpack parameters
    logit_probs = l[:, :, :, :nr_mix]
    l = l[:, :, :, nr_mix:].contiguous().view(xs + [nr_mix * 3])
    # sample mixture indicator from softmax
    temp = torch.FloatTensor(logit_probs.size())
    if l.is_cuda: temp = temp.cuda()
    temp.uniform_(1e-5, 1. - 1e-5)
    temp = logit_probs.data #- torch.log(- torch.log(temp))
    _, argmax = temp.max(dim=3)

    one_hot = to_one_hot(argmax, nr_mix)
    sel = one_hot.view(xs[:-1] + [1, nr_mix])
    # select logistic parameters
    means = torch.sum(l[:, :, :, :, :nr_mix] * sel, dim=4)
    log_scales = torch.clamp(torch.sum(
        l[:, :, :, :, nr_mix:2 * nr_mix] * sel, dim=4), min=-7.)
    coeffs = torch.sum(torch.tanh(
        l[:, :, :, :, 2 * nr_mix:3 * nr_mix]) * sel, dim=4)
    # sample from logistic & clip to interval
    # we don't actually round to the nearest 8bit value when sampling
    u = torch.FloatTensor(means.size())
    if l.is_cuda: u = u.cuda()
    u.uniform_(1e-5, 1. - 1e-5)
    x = means #+ torch.exp(log_scales) * (torch.log(u) - torch.log(1. - u))
    x0 = torch.clamp(torch.clamp(x[:, :, :, 0], min=-1.), max=1.)
    x1 = torch.clamp(torch.clamp(
        x[:, :, :, 1] + coeffs[:, :, :, 0] * x0, min=-1.), max=1.)
    x2 = torch.clamp(torch.clamp(
        x[:, :, :, 2] + coeffs[:, :, :, 1] * x0 + coeffs[:, :, :, 2] * x1, min=-1.), max=1.)

    out = torch.cat([x0.view(xs[:-1] + [1]), x1.view(xs[:-1] + [1]), x2.view(xs[:-1] + [1])], dim=3)
    # put back in Pytorch ordering
    out = out.permute(0, 3, 1, 2)
    return out


def log_logistic_256(x, l):
    """ log-likelihood for mixture of discretized logistics, assumes the data has been rescaled to [-1,1] interval """
    # Pytorch ordering
    x = x.reshape(-1, 3, 64, 64)

    x = x.permute(0, 2, 3, 1)
    l = l.permute(0, 2, 3, 1)
    xs = [int(y) for y in x.size()]
    ls = [int(y) for y in l.size()]

    # here and below: unpacking the params of the mixture of logistics
    nr_mix = int(ls[-1] / 10)
    logit_probs = l[:, :, :, :nr_mix]
    l = l[:, :, :, nr_mix:].contiguous().view(xs + [nr_mix * 3])  # 3 for mean, scale, coef
    means = l[:, :, :, :, :nr_mix]
    # log_scales = torch.max(l[:, :, :, :, nr_mix:2 * nr_mix], -7.)
    log_scales = torch.clamp(l[:, :, :, :, nr_mix:2 * nr_mix], min=-7.)

    coeffs = torch.tanh(l[:, :, :, :, 2 * nr_mix:3 * nr_mix])
    # here and below: getting the means and adjusting them based on preceding
    # sub-pixels
    x = x.contiguous()
    x = x.unsqueeze(-1) + torch.zeros(xs + [nr_mix], device=x.device)
    m2 = (means[:, :, :, 1, :] + coeffs[:, :, :, 0, :]
          * x[:, :, :, 0, :]).view(xs[0], xs[1], xs[2], 1, nr_mix)

    m3 = (means[:, :, :, 2, :] + coeffs[:, :, :, 1, :] * x[:, :, :, 0, :] +
          coeffs[:, :, :, 2, :] * x[:, :, :, 1, :]).view(xs[0], xs[1], xs[2], 1, nr_mix)

    means = torch.cat((means[:, :, :, 0, :].unsqueeze(3), m2, m3), dim=3)
    centered_x = x - means
    inv_stdv = torch.exp(-log_scales)
    plus_in = inv_stdv * (centered_x + 1. / 255.)
    cdf_plus = torch.sigmoid(plus_in)
    min_in = inv_stdv * (centered_x - 1. / 255.)
    cdf_min = torch.sigmoid(min_in)
    # log probability for edge case of 0 (before scaling)
    log_cdf_plus = plus_in - F.softplus(plus_in)
    # log probability for edge case of 255 (before scaling)
    log_one_minus_cdf_min = -F.softplus(min_in)
    cdf_delta = cdf_plus - cdf_min  # probability for all other cases
    mid_in = inv_stdv * centered_x
    # log probability in the center of the bin, to be used in extreme cases
    # (not actually used in our code)
    log_pdf_mid = mid_in - log_scales - 2. * F.softplus(mid_in)
    inner_inner_cond = (cdf_delta > 1e-5).float()
    inner_inner_out = inner_inner_cond * torch.log(torch.clamp(cdf_delta, min=1e-12)) + (1. - inner_inner_cond) * (
            log_pdf_mid - np.log(127.5))
    inner_cond = (x > 0.999).float()
    inner_out = inner_cond * log_one_minus_cdf_min + (1. - inner_cond) * inner_inner_out
    cond = (x < -0.999).float()
    log_probs = cond * log_cdf_plus + (1. - cond) * inner_out
    log_probs = torch.sum(log_probs, dim=3) + torch.log_softmax(logit_probs, dim=-1)
    log_probs = torch.logsumexp(log_probs, dim=-1)

    return log_probs.sum((1, 2))

# def log_logistic_256(x, mean, logvar, average=False, reduce=True, dim=1):
#    # print(x.shape)
#    # print(mean.shape)
#    bin_size = 1. / 256.
#     # implementation like https://github.com/openai/iaf/blob/master/tf_utils/distributions.py#L28
#    x_lower = torch.floor(x / bin_size)
#    scale = torch.exp(logvar)
#    x = (torch.floor(x / bin_size) * bin_size - mean) / scale
#    cdf_minus = torch.sigmoid(x)
#    cdf_plus = torch.sigmoid(x + bin_size/scale)
#    log_cdf_rest = torch.log(cdf_plus - cdf_minus + 1e-8)
#    mask_255 = (x_lower == 255.).float()
#    mask_0 = (x_lower == 0.).float()
#    log_cdf_0 = x+bin_size/scale - torch.nn.functional.softplus(x+bin_size/scale)
#    log_cdf_255 = -torch.nn.functional.softplus(x)
#    log_logist_256 = (1-mask_0-mask_255)*log_cdf_rest + mask_255*log_cdf_255 + mask_0*log_cdf_0
#    if average:
#        return torch.mean(log_logist_256, dim)
#    else:
#        return torch.sum(log_logist_256, dim)

#def log_logistic_256(x, mean, logvar, average=False, reduce=True, dim=None):
#    bin_size = 1. / 256.
    # implementation like https://github.com/openai/iaf/blob/master/tf_utils/distributions.py#L28
#    x_lower = torch.floor(x / bin_size)
#    scale = torch.exp(logvar)
#    x = (torch.floor(x / bin_size) * bin_size - mean) / scale
#    cdf_plus = torch.sigmoid(x + bin_size/scale)
#    mask = (x_lower == 255.)
#    cdf_plus_masked = cdf_plus.masked_fill(value=1., mask=mask)
#    cdf_minus = torch.sigmoid(x)
#    mask = (x_lower == 0.)
#    cdf_minus_masked = cdf_minus.masked_fill(value=0., mask=mask)
#    log_logist_256 = torch.log(cdf_plus_masked - cdf_minus_masked + 1e-7)
#    if average:
#        return torch.mean(log_logist_256, dim)
#    else:
#        return torch.sum(log_logist_256, dim)

