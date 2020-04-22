from __future__ import print_function
import numpy as np
import torch
import torch.utils.data
import torch.nn as nn
from torch.nn import Linear
from utils.nn import GatedDense, NonLinear
from models.AbsHDownModel import AbsHDownModel


class VAE(AbsHDownModel):
    def __init__(self, args):
        super(VAE, self).__init__(args)

    def create_model(self, args):
        print("create_model")

        # becasue super is using h_size
        self.args = args

        # encoder: q(z2 | x)
        self.q_z_layers = nn.Sequential(
            GatedDense(np.prod(self.args.input_size), self.args.hidden_size),
            GatedDense(self.args.hidden_size, self.args.hidden_size)
        )

        self.q_z_mean = Linear(self.args.hidden_size, self.args.z2_size)

        if args.same_variational_var:
            self.q_z_logvar = torch.nn.Parameter(torch.randn((1)))
        else:
            self.q_z_logvar = NonLinear(self.args.hidden_size, self.args.z2_size, activation=nn.Hardtanh(min_val=-6., max_val=2.))

        # encoder: q(z1 | x, z2)
        self.q_z1_layers_x = nn.Sequential(
            GatedDense(np.prod(self.args.input_size), self.args.hidden_size)
        )
        self.q_z1_layers_z2 = nn.Sequential(
            GatedDense(self.args.z2_size, self.args.hidden_size)
        )
        self.q_z1_layers_joint = nn.Sequential(
            GatedDense(2 * self.args.hidden_size, self.args.hidden_size)
        )

        self.q_z1_mean = Linear(self.args.hidden_size, self.args.z1_size)
        self.q_z1_logvar = NonLinear(self.args.hidden_size, self.args.z1_size, activation=nn.Hardtanh(min_val=-6., max_val=2.))

        # decoder: p(z1 | z2)
        self.p_z1_layers_z2 = nn.Sequential(
            GatedDense(self.args.z2_size, self.args.hidden_size),
            GatedDense(self.args.hidden_size, self.args.hidden_size)
        )

        self.p_z1_mean = Linear(self.args.hidden_size, self.args.z1_size)
        self.p_z1_logvar = NonLinear(self.args.hidden_size, self.args.z1_size, activation=nn.Hardtanh(min_val=-6.,max_val=2.))

        # decoder: p(x | z1, z2)
        self.p_x_layers_z1 = nn.Sequential(
            GatedDense(self.args.z1_size, self.args.hidden_size)
        )
        self.p_x_layers_z2 = nn.Sequential(
            GatedDense(self.args.z2_size, self.args.hidden_size)
        )
        self.p_x_layers_joint = nn.Sequential(
            GatedDense(2 * self.args.hidden_size, self.args.hidden_size)
        )


