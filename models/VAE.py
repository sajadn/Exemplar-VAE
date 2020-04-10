from __future__ import print_function
import numpy as np
import torch
import torch.utils.data
import torch.nn as nn
from torch.nn import Linear
from utils.nn import GatedDense, NonLinear
from models.AbsModel import AbsModel


class VAE(AbsModel):
    def __init__(self, args):
        super(VAE, self).__init__(args)

    def create_model(self, args, train_data_size=None):
        self.train_data_size = train_data_size
        self.q_z_layers = nn.Sequential(
            GatedDense(np.prod(self.args.input_size), self.args.hidden_size, no_attention=self.args.no_attention),
            GatedDense(self.args.hidden_size, self.args.hidden_size, no_attention=self.args.no_attention)
        )
        self.q_z_mean = Linear(self.args.hidden_size, self.args.z1_size)
        if args.same_variational_var:
            self.q_z_logvar = torch.nn.Parameter(torch.randn((1)))
        else:
            self.q_z_logvar = NonLinear(self.args.hidden_size,
                                        self.args.z1_size, activation=nn.Hardtanh(min_val=-6., max_val=2.))

        self.p_x_layers = nn.Sequential(
            GatedDense(self.args.z1_size, self.args.hidden_size, no_attention=self.args.no_attention),
            GatedDense(self.args.hidden_size, self.args.hidden_size, no_attention=self.args.no_attention))



