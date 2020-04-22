from __future__ import print_function
import numpy as np
import torch.nn as nn
from utils.nn import  GatedDense, NonLinear, \
    Conv2d, GatedConv2d
from models.AbsHDownModel import AbsHDownModel


class VAE(AbsHDownModel):
    def __init__(self, args):
        super(VAE, self).__init__(args)

    def create_model(self, args):
        if args.dataset_name == 'freyfaces':
            self.h_size = 210
        elif args.dataset_name == 'cifar10' or args.dataset_name == 'svhn':
            self.h_size = 384
        else:
            self.h_size = 294

        fc_size = 300

        # encoder: q(z2 | x)
        self.q_z_layers = nn.Sequential(
            GatedConv2d(self.args.input_size[0], 32, 7, 1, 3, no_attention=args.no_attention),
            GatedConv2d(32, 32, 3, 2, 1, no_attention=args.no_attention),
            GatedConv2d(32, 64, 5, 1, 2, no_attention=args.no_attention),
            GatedConv2d(64, 64, 3, 2, 1, no_attention=args.no_attention),
            GatedConv2d(64, 6, 3, 1, 1, no_attention=args.no_attention)
        )

        # linear layers
        self.q_z_mean = NonLinear(self.h_size, self.args.z2_size, activation=None)

        # SAME VARAITIONAL VAR TO SEE IF IT HELPS
        self.q_z_logvar = NonLinear(self.h_size, self.args.z2_size, activation=nn.Hardtanh(min_val=-6., max_val=2.))

        # encoder: q(z1|x,z2)
        # PROCESSING x
        self.q_z1_layers_x = nn.Sequential(
            GatedConv2d(self.args.input_size[0], 32, 3, 1, 1, no_attention=args.no_attention),
            GatedConv2d(32, 32, 3, 2, 1, no_attention=args.no_attention),
            GatedConv2d(32, 64, 3, 1, 1, no_attention=args.no_attention),
            GatedConv2d(64, 64, 3, 2, 1, no_attention=args.no_attention),
            GatedConv2d(64, 6, 3, 1, 1, no_attention=args.no_attention)
        )
        # PROCESSING Z2
        self.q_z1_layers_z2 = nn.Sequential(GatedDense(self.args.z2_size, self.h_size))

        # PROCESSING JOINT
        self.q_z1_layers_joint = nn.Sequential(GatedDense(2* self.h_size, fc_size))

        # linear layers
        self.q_z1_mean = NonLinear(fc_size, self.args.z1_size, activation=None)
        self.q_z1_logvar = NonLinear(fc_size, self.args.z1_size, activation=nn.Hardtanh(min_val=-6., max_val=2.))

        # decoder p(z1|z2)
        self.p_z1_layers_z2 = nn.Sequential(
            GatedDense(self.args.z2_size, fc_size, no_attention=args.no_attention),
            GatedDense(fc_size, fc_size, no_attention=args.no_attention)
        )
        self.p_z1_mean = NonLinear(fc_size, self.args.z1_size, activation=None)
        self.p_z1_logvar = NonLinear(fc_size, self.args.z1_size, activation=nn.Hardtanh(min_val=-6., max_val=2.))

        # decoder: p(x | z)
        self.p_x_layers_z1 = nn.Sequential(
            GatedDense(self.args.z1_size, fc_size, no_attention=args.no_attention)
        )
        self.p_x_layers_z2 = nn.Sequential(
            GatedDense(self.args.z2_size, fc_size, no_attention=args.no_attention)
        )

        self.p_x_layers_joint_pre = nn.Sequential(
            GatedDense(2 * fc_size, np.prod(self.args.input_size), no_attention=args.no_attention)
        )

        # decoder: p(x | z)
        self.p_x_layers_joint = nn.Sequential(
            GatedConv2d(self.args.input_size[0], 64, 3, 1, 1, no_attention=args.no_attention),
            GatedConv2d(64, 64, 3, 1, 1, no_attention=args.no_attention),
            GatedConv2d(64, 64, 3, 1, 1, no_attention=args.no_attention),
            GatedConv2d(64, 64, 3, 1, 1, no_attention=args.no_attention),
        )

        if self.args.input_type == 'binary':
            self.p_x_mean = Conv2d(64, 1, 1, 1, 0, activation=nn.Sigmoid())
        elif self.args.input_type == 'gray' or self.args.input_type == 'continuous':
            self.p_x_mean = Conv2d(64, self.args.input_size[0], 1, 1, 0)
            self.p_x_logvar = Conv2d(64, self.args.input_size[0], 1, 1, 0, activation=nn.Hardtanh(min_val=-4.5, max_val=0.))
        elif self.args.input_type == 'pca':
            self.p_x_mean = Conv2d(64, 1, 1, 1, 0)
            self.p_x_logvar = Conv2d(64, self.args.input_size[0], 1, 1, 0, activation=nn.Hardtanh(min_val=-4.5, max_val=0.))

    # THE MODEL: FORWARD PASS
    def forward(self, x):
        x = x.view(-1, self.args.input_size[0], self.args.input_size[1], self.args.input_size[2])
        return super(VAE, self).forward(x)

