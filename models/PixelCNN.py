from __future__ import print_function
import numpy as np
import torch.nn as nn
from utils.nn import  GatedDense, NonLinear, \
    Conv2d, GatedConv2d, MaskedConv2d, PixelSNAIL
from models.AbsHModel import BaseHModel
import torch

class VAE(BaseHModel):
    def __init__(self, args):
        super(VAE, self).__init__(args)

    def create_model(self, args):
        if args.dataset_name == 'freyfaces':
            self.h_size = 210
        elif args.dataset_name == 'cifar10' or args.dataset_name == 'svhn':
            self.h_size = 384
        else:
            self.h_size = 294

        # encoder: q(z2 | x)
        self.q_z_layers = nn.Sequential(
            GatedConv2d(self.args.input_size[0], 32, 7, 1, 3),
            GatedConv2d(32, 32, 3, 2, 1),
            GatedConv2d(32, 64, 5, 1, 2),
            GatedConv2d(64, 64, 3, 2, 1),
            GatedConv2d(64, 6, 3, 1, 1)
        )
        # linear layers
        self.q_z_mean = NonLinear(self.h_size, self.args.z2_size, activation=None)
        self.q_z_logvar = NonLinear(self.h_size, self.args.z2_size, activation=nn.Hardtanh(min_val=-6., max_val=2.))

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
            GatedDense(self.args.z2_size, self.h_size)
        )
        # PROCESSING JOINT
        self.q_z1_layers_joint = nn.Sequential(
            GatedDense( 2 * self.h_size, 300)
        )
        # linear layers
        self.q_z1_mean = NonLinear(300, self.args.z1_size, activation=None)
        self.q_z1_logvar = NonLinear(300, self.args.z1_size, activation=nn.Hardtanh(min_val=-6., max_val=2.))

        # decoder p(z1|z2)
        self.p_z1_layers_z2 = nn.Sequential(
            GatedDense(self.args.z2_size, 300),
            GatedDense(300, 300)
        )
        self.p_z1_mean = NonLinear(300, self.args.z1_size, activation=None)
        self.p_z1_logvar = NonLinear(300, self.args.z1_size, activation=nn.Hardtanh(min_val=-6., max_val=2.))

        # decoder: p(x | z)
        self.p_x_layers_z1 = nn.Sequential(
            GatedDense(self.args.z1_size, np.prod(self.args.input_size))
        )
        self.p_x_layers_z2 = nn.Sequential(
            GatedDense(self.args.z2_size, np.prod(self.args.input_size))
        )

        # decoder: p(x | z)
        act = nn.ReLU(True)
        #self.pixelcnn = nn.Sequential(
        #    MaskedConv2d('A', self.args.input_size[0] + 2 * self.args.input_size[0], 64, 3, 1, 1, bias=False),
        #    nn.BatchNorm2d(64), act,
        #    MaskedConv2d('B', 64, 64, 3, 1, 1, bias=False), nn.BatchNorm2d(64), act,
        #    MaskedConv2d('B', 64, 64, 3, 1, 1, bias=False), nn.BatchNorm2d(64), act,
        #    MaskedConv2d('B', 64, 64, 3, 1, 1, bias=False), nn.BatchNorm2d(64), act,
        #    MaskedConv2d('B', 64, 64, 3, 1, 1, bias=False), nn.BatchNorm2d(64), act,
        #    MaskedConv2d('B', 64, 64, 3, 1, 1, bias=False), nn.BatchNorm2d(64), act,
        #    MaskedConv2d('B', 64, 64, 3, 1, 1, bias=False), nn.BatchNorm2d(64), act,
        #    MaskedConv2d('B', 64, 64, 3, 1, 1, bias=False), nn.BatchNorm2d(64), act
        #)
        self.pixelcnn = PixelSNAIL([32, 32], 32, 32, 3, 1, 4, 32)

        if self.args.input_type == 'binary':
            self.p_x_mean = Conv2d(64, 1, 1, 1, 0, activation=nn.Sigmoid())
        elif self.args.input_type == 'gray' or self.args.input_type == 'continuous':
            self.p_x_mean = Conv2d(32, 100, 1, 1, 0, bias=False)
            # self.p_x_logvar = Conv2d(64, self.args.input_size[0], 1, 1, 0, activation=nn.Hardtanh(min_val=-4.5, max_val=0.), bias=False)

    def pixelcnn_generate(self, z1, z2):
        # Sampling from PixelCNN
        x_zeros = torch.zeros(
            (z1.size(0), self.args.input_size[0], self.args.input_size[1], self.args.input_size[2]))
        x_zeros = x_zeros.to(self.args.device)
        for i in range(self.args.input_size[1]):
            for j in range(self.args.input_size[2]):
                samples_mean, samples_logvar = self.p_x(z1, z2, x=x_zeros.detach())
                samples_mean = samples_mean.view(samples_mean.size(0), 100, self.args.input_size[1],
                                                 self.args.input_size[2])
                from utils.distributions import sample_from_discretized_mix_logistic
                if self.args.input_type == 'binary':
                    probs = samples_mean[:, :, i, j].data
                    x_zeros[:, :, i, j] = torch.bernoulli(probs).float()
                    samples_gen = samples_mean

                elif self.args.input_type == 'gray' or self.args.input_type == 'continuous':
                    sample = sample_from_discretized_mix_logistic(samples_mean, nr_mix=10)
                    samples_gen = sample
                    x_zeros[:, :, i, j] = sample[:, :, i, j]
        return samples_gen

    def forward(self, x):
        x = x.view(-1, self.args.input_size[0], self.args.input_size[1], self.args.input_size[2])
        return super(VAE, self).forward(x)


