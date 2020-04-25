from __future__ import print_function
import numpy as np
import torch
import torch.utils.data
from utils.distributions import log_normal_diag
from .BaseModel import BaseModel
from utils.utils import reparameterize


class AbsUpModel(BaseModel):
    def __init__(self, args):
        super(AbsUpModel, self).__init__(args)

    def generate_x_from_z(self, z, with_reparameterize=True):
        z1_sample_mean, z1_sample_logvar, h = self.p_z1(z)
        if with_reparameterize:
            z1_sample_rand = reparameterize(z1_sample_mean, z1_sample_logvar)
        else:
            z1_sample_rand = z1_sample_mean

        generated_xs, _ = self.p_x(z1_sample_rand.view(-1, self.args.z1_size), h)
        return generated_xs

    def p_z1(self, z2):
        if 'conv' in self.args.model_name:
            z2 = z2.reshape(-1, self.bottleneck, self.args.input_size[1] // 2, self.args.input_size[1] // 2)
        h = self.p_z1_layers_z2(z2)
        z1_p_mean = self.p_z1_mean(h)
        z1_p_logvar = self.p_z1_logvar(h)
        return z1_p_mean, z1_p_logvar, h

    def q_z1(self, x):
        h = self.q_z1_layers_x(x)
        z1_q_mean = self.q_z1_mean(h)
        z1_q_logvar = self.q_z1_logvar(h)
        return z1_q_mean, z1_q_logvar, h

    def p_x(self, z1, h):
        if 'conv' in self.args.model_name:
            z1 = z1.reshape(-1, self.bottleneck, self.args.input_size[1] // 2, self.args.input_size[1] // 2)
            h = self.p_x_first_z1(h)
        h = torch.cat((h, z1), 1)
        h = self.p_x_layers_z1(h)
        x_mean = self.p_x_mean(h)
        if self.args.use_logit is False:
            x_mean = torch.clamp(x_mean, min=0. + 1. / 512., max=1. - 1. / 512.)
        reshaped_var = self.decoder_logstd.unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
        x_logvar = reshaped_var * x_mean.new_ones(size=x_mean.shape)
        return x_mean.reshape(-1, np.prod(self.args.input_size)), x_logvar.reshape(-1, np.prod(self.args.input_size))

    def q_z(self, x=None, z1=None, prior=False):
        if z1 is None:
            x = x.view(-1, *self.args.input_size)
            z1, _, x = self.q_z1(x)
        x = self.q_z_first_layer(x)
        x = torch.cat((x, z1), dim=1)
        x = self.q_z_layers(x)
        z_q_mean = self.q_z_mean(x)
        if prior is True:
            if self.args.prior == 'exemplar_prior':
                z_q_logvar = self.prior_log_variance * torch.ones((x.shape[0], self.args.z1_size)).to(self.args.device)
            else:
                z_q_logvar = self.q_z_logvar(x)
        else:
            z_q_logvar = self.q_z_logvar(x)
        return z_q_mean.reshape(-1, self.args.z1_size), z_q_logvar.reshape(-1, self.args.z1_size)

    def forward(self, x):
        x = x.view(-1, *self.args.input_size)
        z1_q_mean, z1_q_logvar, h = self.q_z1(x)
        z1_q = reparameterize(z1_q_mean, z1_q_logvar)
        z2_q_mean, z2_q_logvar = self.q_z(h, z1_q)
        z2_q = reparameterize(z2_q_mean, z2_q_logvar)
        z1_p_mean, z1_p_logvar, h = self.p_z1(z2_q)
        x_mean, x_logvar = self.p_x(z1_q, h)
        z_stats = [(z1_q, z1_q_mean, z1_q_logvar, z1_p_mean, z1_p_logvar), (z2_q, z2_q_mean, z2_q_logvar)]
        return x_mean, x_logvar, z_stats
