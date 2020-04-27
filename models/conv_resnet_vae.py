from __future__ import print_function
import torch
import torch.utils.data
import torch.nn as nn
from models.AbsModel import AbsModel
from torch.nn.utils import weight_norm
from utils.utils import reparameterize
import numpy as np
from utils.utils import inverse_scaled_logit


class block(nn.Module):
    def __init__(self, input_size, output_size, stride=1, kernel=3, padding=1, bottleneck=32, resnet_coeff=1.):
        super(block, self).__init__()
        self.bottleneck = bottleneck
        self.resnet_coeff = resnet_coeff
        self.conv1_forward = weight_norm(
            nn.Conv2d(input_size, output_size + self.bottleneck, kernel_size=kernel, stride=stride,
                      padding=padding,
                      bias=True))
        self.conv2_forward = weight_norm(
            nn.Conv2d(output_size, output_size, kernel_size=kernel, stride=stride, padding=padding,
                      bias=True))
        self.conv1_backward = weight_norm(
            nn.Conv2d(input_size, output_size + self.bottleneck, kernel_size=kernel, stride=stride,
                      padding=padding,
                      bias=True))
        self.conv2_backward = weight_norm(
            nn.Conv2d(output_size, output_size, kernel_size=kernel, stride=stride, padding=padding,
                      bias=True))
        self.activation = torch.nn.ELU()
        self.q_z = None

    def forward(self, x, prior=False):
        out1 = self.conv1_forward(self.activation(x))
        q_z_mean = out1[:, -2 * self.bottleneck:-self.bottleneck, :, :]
        q_z_logvar = out1[:, -self.bottleneck:, :, :]
        if prior is False:
            q_z = reparameterize(q_z_mean, q_z_logvar)
            self.q_z = q_z
        else:
            q_z = q_z_mean

        return x+self.resnet_coeff*self.conv2_forward(self.activation(torch.cat((out1[:, :-2 * self.bottleneck, :, :],
                                                             q_z),
                                                  dim=1))), (q_z, q_z_mean, q_z_logvar)

    def backward(self, x):
        out1 = self.conv1_backward(self.activation(x))
        p_z_mean = out1[:, -2 * self.bottleneck:-self.bottleneck, :, :]
        p_z_logvar = out1[:, -self.bottleneck:, :, :]
        if self.q_z is not None:
            out = x + self.resnet_coeff*self.conv2_backward(
                self.activation(torch.cat((out1[:, :-2 * self.bottleneck, :, :], self.q_z), dim=1))), \
                   (p_z_mean, p_z_logvar)
            self.q_z = None
            return out
        else:
            return x + self.resnet_coeff*self.conv2_backward(
                self.activation(torch.cat((out1[:, :-2 * self.bottleneck, :, :], reparameterize(p_z_mean, p_z_logvar)), dim=1))), \
                            (p_z_mean, p_z_logvar)


class VAE(AbsModel):
    def __init__(self, args):
        super(VAE, self).__init__(args)

    def create_model(self, args, train_data_size=None):

        self.train_data_size = train_data_size
        self.cs = 100
        self.bottleneck=self.args.bottleneck

        self.down_sample = nn.Sequential(
                            nn.ELU(),
                            weight_norm(nn.Conv2d(in_channels=self.args.input_size[0],
                                  out_channels=self.cs, kernel_size=3, stride=2, padding=1)))

        self.up_sample = nn.Upsample(scale_factor=2)

        self.layers = nn.ModuleList([
            *[block(input_size=self.cs, output_size=self.cs, stride=1, kernel=3, padding=1, bottleneck=self.bottleneck,
                    resnet_coeff=args.resnet_coeff)
              for _ in range(self.args.rs_blocks)]])

        self.q_z_mean = nn.Sequential(nn.ELU(),
                                      weight_norm(nn.Conv2d(in_channels=self.cs,
                                                            out_channels=self.bottleneck,
                                                            kernel_size=3,
                                                            stride=1, padding=1)))
        self.q_z_logvar = nn.Sequential(nn.ELU(),
                                      weight_norm(nn.Conv2d(in_channels=self.cs,
                                                            out_channels=self.bottleneck,
                                                            kernel_size=3,
                                                            stride=1, padding=1)))

        self.p_x_mean = nn.Sequential(nn.ELU(),
                                      weight_norm(nn.Conv2d(in_channels=self.cs,
                                                            out_channels=self.args.input_size[0],
                                                            kernel_size=3, stride=1, padding=1)))
        self.p_x_first = nn.Sequential(nn.ELU(),
                            weight_norm(nn.Conv2d(in_channels=self.bottleneck,
                                            out_channels=self.cs,
                                            kernel_size=3,
                                            stride=1, padding=1)))

    def generate_x_from_z(self, z, with_reparameterize=True):
        generated_x, _, _ = self.p_x(z)
        try:
            if self.args.use_logit is True:
                return torch.floor(inverse_scaled_logit(generated_x, self.args.lambd)*256).int()
            else:
                return generated_x
        except Exception as e:
            print(e)
            return generated_x

    def q_z_layers(self, x, prior=False):
        x = self.down_sample(x)
        qs = []
        for l in self.layers:
            x, q_stat = l(x, prior=prior)
            qs.append(q_stat)
        return x, qs

    def p_x_layers(self, x):
        ps = []
        x = self.p_x_first(x)
        for l in self.layers[::-1]:
            x, p_stat = l.backward(x)
            ps.append(p_stat)
        x = self.up_sample(x)
        return x, ps

    def p_x(self, z):
        if 'conv' in self.args.model_name:
            z = z.reshape(-1, self.bottleneck, self.args.input_size[1]//2, self.args.input_size[1]//2)
        z, ps = self.p_x_layers(z)
        x_mean = self.args.resnet_coeff*self.p_x_mean(z)
        if self.args.input_type == 'binary':
            x_logvar = torch.zeros(1, np.prod(self.args.input_size))
        else:
            if self.args.use_logit is False:
                if self.args.zero_center:
                    x_mean += 0.5
                x_mean = torch.clamp(x_mean, min=0.+1./512., max=1.-1./512.)
            reshaped_var = self.decoder_logstd.unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
            x_logvar = reshaped_var*x_mean.new_ones(size=x_mean.shape)
        return x_mean.reshape(-1, np.prod(self.args.input_size)),\
               x_logvar.reshape(-1, np.prod(self.args.input_size)), ps

    def q_z(self, x, z1=None, prior=False):
        if 'conv' in self.args.model_name or self.args.model_name=='CelebA':
            x = x.view(-1, *self.args.input_size)
        h, qs = self.q_z_layers(x, prior=prior)
        if self.args.model_name == 'convhvae_2level':
            h = h.view(x.size(0), -1)
        z_q_mean = self.q_z_mean(h)
        if prior is True:
            if self.args.prior == 'exemplar_prior':
                z_q_logvar = self.prior_log_variance * torch.ones((x.shape[0], self.args.z1_size)).to(self.args.device)
            else:
                z_q_logvar = self.q_z_logvar(h)
            return z_q_mean.reshape(-1, self.args.z1_size), z_q_logvar.reshape(-1, self.args.z1_size)
        else:
            z_q_logvar = self.q_z_logvar(h)
            return z_q_mean.reshape(-1, self.args.z1_size), z_q_logvar.reshape(-1, self.args.z1_size), qs

    def forward(self, x):
        z_q_mean, z_q_logvar, qs = self.q_z(x)
        z_q = reparameterize(z_q_mean, z_q_logvar)
        x_mean, x_logvar, ps = self.p_x(z_q)

        z_stats = []
        for i in range(len(qs)):
            z_stats.append((*qs[i], *ps.pop()))

        z_stats.append((z_q, z_q_mean, z_q_logvar))
        return x_mean, x_logvar, z_stats
