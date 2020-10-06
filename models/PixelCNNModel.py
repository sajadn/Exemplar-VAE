from __future__ import print_function
import numpy as np
import torch.nn as nn
from utils.nn import  GatedDense, NonLinear, \
    Conv2d, GatedConv2d
from models.AbsHModel import BaseHModel
import torch

class VAE(BaseHModel):
    def __init__(self, args):
        super(VAE, self).__init__(args)

        if self.args.dataset_name == 'freyfaces':
            h_size = 210
        elif self.args.dataset_name == 'cifar10':
            h_size = 384
        else:
            h_size = 294

        # encoder: q(z2 | x)
        self.q_z2_layers = nn.Sequential(
            GatedConv2d(self.args.input_size[0], 32, 7, 1, 3),
            GatedConv2d(32, 32, 3, 2, 1),
            GatedConv2d(32, 64, 5, 1, 2),
            GatedConv2d(64, 64, 3, 2, 1),
            GatedConv2d(64, 6, 3, 1, 1)
        )
        # linear layers
        self.q_z2_mean = NonLinear(h_size, self.args.z2_size, activation=None)
        self.q_z2_logvar = NonLinear(h_size, self.args.z2_size, activation=nn.Hardtanh(min_val=-6., max_val=2.))

        # encoder: q(z1|x,z2)
        # PROCESSING x
        self.q_z1_layers_x = nn.Sequential(
            GatedConv2d(self.args.input_size[0], 32, 3, 1, 1),
            GatedConv2d(32, 32, 3, 2, 1),
            GatedConv2d(32, 64, 3, 1, 1),
            GatedConv2d(64, 64, 3, 2, 1),
            GatedConv2d(64, 6, 3, 1, 1)
        )
        # PROCESSING Z2
        self.q_z1_layers_z2 = nn.Sequential(
            GatedDense(self.args.z2_size, h_size)
        )
        # PROCESSING JOINT
        self.q_z1_layers_joint = nn.Sequential(
            GatedDense( 2 * h_size, 300)
        )
        # linear layers
        self.q_z1_mean = NonLinear(300, self.args.z1_size, activation=None)
        self.q_z1_logvar = NonLinear(300, self.args.z1_size, activation=nn.Hardtanh(min_val=-6., max_val=2.))

        # decoder p(z1|z2)
        self.p_z1_layers = nn.Sequential(
            GatedDense(self.args.z2_size, 300),
            GatedDense(300, 300)
        )
        self.p_z1_mean = NonLinear(300, self.args.z1_size, activation=None)
        self.p_z1_logvar = NonLinear(300, self.args.z1_size, activation=nn.Hardtanh(min_val=-6., max_val=2.))

        # decoder: p(x | z)
        self.p_x_layers_z1 = nn.Sequential(
            GatedDense(self.args.z1_size, 300)
        )
        self.p_x_layers_z2 = nn.Sequential(
            GatedDense(self.args.z2_size, 300)
        )

        self.p_x_layers_joint_pre = nn.Sequential(
            GatedDense(2 * 300, np.prod(self.args.input_size))
        )

        # decoder: p(x | z)
        act = nn.ReLU(True)
        # joint
        self.p_x_layers_joint = nn.Sequential(
            GatedConv2d(self.args.input_size[0], 64, 3, 1, 1),
            GatedConv2d(64, 64, 3, 1, 1),
            GatedConv2d(64, 64, 3, 1, 1),
            GatedConv2d(64, 64, 3, 1, 1),
        )

        if self.args.input_type == 'binary':
            self.p_x_mean = Conv2d(64, 1, 1, 1, 0, activation=nn.Sigmoid())
        elif self.args.input_type == 'gray' or self.args.input_type == 'continuous':
            self.p_x_mean = Conv2d(64, self.args.input_size[0], 1, 1, 0, activation=nn.Sigmoid())
            self.p_x_logvar = Conv2d(64, self.args.input_size[0], 1, 1, 0, activation=nn.Hardtanh(min_val=-4.5, max_val=0.))

    def pixelcnn_generate(self, z1, z2):
        # Sampling from PixelCNN
        x_zeros = torch.zeros(
            (z1.size(0), self.args.input_size[0], self.args.input_size[1], self.args.input_size[2]))
        if self.args.cuda:
            x_zeros = x_zeros.cuda()

        for i in range(self.args.input_size[1]):
            for j in range(self.args.input_size[2]):
                samples_mean, samples_logvar = self.p_x(x_zeros.detach(), z1, z2)
                samples_mean = samples_mean.view(samples_mean.size(0), self.args.input_size[0], self.args.input_size[1],
                                                 self.args.input_size[2])

                if self.args.input_type == 'binary':
                    probs = samples_mean[:, :, i, j].data
                    x_zeros[:, :, i, j] = torch.bernoulli(probs).float()
                    samples_gen = samples_mean

                elif self.args.input_type == 'gray' or self.args.input_type == 'continuous':
                    binsize = 1. / 256.
                    samples_logvar = samples_logvar.view(samples_mean.size(0), self.args.input_size[0],
                                                         self.args.input_size[1], self.args.input_size[2])
                    means = samples_mean[:, :, i, j].data
                    logvar = samples_logvar[:, :, i, j].data
                    # sample from logistic distribution
                    u = torch.rand(means.size()).cuda()
                    y = torch.log(u) - torch.log(1. - u)
                    sample = means + torch.exp(logvar) * y
                    x_zeros[:, :, i, j] = torch.floor(sample / binsize) * binsize
                    samples_gen = samples_mean
        return samples_gen