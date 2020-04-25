from __future__ import print_function
import numpy as np
import torch
import torch.utils.data
from utils.distributions import log_normal_diag
from .BaseModel import BaseModel
from utils.utils import reparameterize


class AbsHDownModel(BaseModel):
    def __init__(self, args):
        super(AbsHDownModel, self).__init__(args)

    def generate_x_from_z(self, z, with_reparameterize=True):
        z1_sample_mean, z1_sample_logvar = self.p_z1(z)
        if with_reparameterize:
            z1_sample_rand = reparameterize(z1_sample_mean, z1_sample_logvar)
        else:
            z1_sample_rand = z1_sample_mean

        generated_xs, _ = self.p_x(z1_sample_rand.view(-1, self.args.z1_size),
                                   z.view(-1, self.args.z2_size))
        return generated_xs

    def p_z1(self, z2):
        z2 = self.p_z1_layers_z2(z2)
        z1_p_mean = self.p_z1_mean(z2)
        z1_p_logvar = self.p_z1_logvar(z2)
        return z1_p_mean, z1_p_logvar

    def q_z1(self, x, z):
        x = self.q_z1_layers_x(x)
        if self.args.model_name == 'convhvae_2level':
            x = x.view(x.size(0),-1)
        z2 = self.q_z1_layers_z2(z)
        h = torch.cat((x,z2),1)
        h = self.q_z1_layers_joint(h)
        z1_q_mean = self.q_z1_mean(h)
        z1_q_logvar = self.q_z1_logvar(h)
        return z1_q_mean, z1_q_logvar

    def p_x(self, z1, z2):
        z1 = self.p_x_layers_z1(z1)
        z2 = self.p_x_layers_z2(z2)
        h = torch.cat((z1, z2), 1)
        if 'convhvae_2level' in self.args.model_name:
            h = self.p_x_layers_joint_pre(h)
            h = h.view(-1, self.args.input_size[0], self.args.input_size[1], self.args.input_size[2])
        h_decoder = self.p_x_layers_joint(h)
        x_mean = self.p_x_mean(h_decoder)

        if 'convhvae_2level' in self.args.model_name:
            x_mean = x_mean.view(-1, np.prod(self.args.input_size))

        if self.args.input_type == 'binary':
            x_logvar = 0.
        else:
            x_mean = torch.clamp(x_mean, min=0.+1./512., max=1.-1./512.)
            x_logvar = self.p_x_logvar(h_decoder)
            if 'convhvae_2level' in self.args.model_name:
                x_logvar = x_logvar.view(-1, np.prod(self.args.input_size))
        return x_mean, x_logvar


    def forward(self, x):
        z2_q_mean, z2_q_logvar = self.q_z(x)
        z2_q = reparameterize(z2_q_mean, z2_q_logvar)
        z1_q_mean, z1_q_logvar = self.q_z1(x, z2_q)
        z1_q = reparameterize(z1_q_mean, z1_q_logvar)
        z1_p_mean, z1_p_logvar = self.p_z1(z2_q)
        x_mean, x_logvar = self.p_x(z1_q, z2_q)
        z_stats = [(z1_q, z1_q_mean, z1_q_logvar, z1_p_mean, z1_p_logvar), (z2_q, z2_q_mean, z2_q_logvar)]
        return x_mean, x_logvar, z_stats

