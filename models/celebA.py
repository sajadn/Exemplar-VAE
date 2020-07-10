from __future__ import print_function
import torch
import torch.utils.data
import torch.nn as nn
from models.AbsModel import AbsModel
from torch.nn.utils import weight_norm

class Flatten(torch.nn.Module):
    def forward(self, x):
        batch_size = x.shape[0]
        return x.view(batch_size, -1)

class UnFlatten(torch.nn.Module):
    def __init__(self, size):
        super(UnFlatten, self).__init__()
        self.size = size

    def forward(self, x):
        batch_size = x.shape[0]
        return x.view(batch_size, *self.size)

class VAE(AbsModel):
    def __init__(self, args):
        super(VAE, self).__init__(args)

    def create_model(self, args, train_data_size=None):
        self.train_data_size = train_data_size

        d_size = args.divide_size
        self.q_z_layers = nn.Sequential(
            nn.Conv2d(in_channels=self.args.input_size[0], out_channels=128//d_size, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128//d_size),
            nn.ELU(),
            nn.Conv2d(in_channels=128//d_size, out_channels=256//d_size, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(256//d_size),
            nn.ELU(),
            nn.Conv2d(in_channels=256//d_size, out_channels=512//d_size, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(512//d_size),
            nn.ELU(),
            nn.Conv2d(in_channels=512//d_size, out_channels=1024//d_size, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(1024//d_size),
            nn.ELU(),
            Flatten()
        )

        self.q_z_mean = nn.Sequential(nn.Linear(1024//d_size * 4 * 4, self.args.z1_size),)
        self.q_z_logvar = nn.Sequential(nn.Linear(1024//d_size * 4 * 4, self.args.z1_size))
        self.p_x_layers = nn.Sequential(
            nn.Linear(self.args.z1_size, 1024//d_size * 8*8),
            UnFlatten(size=[1024 // d_size, 8, 8]),
            nn.BatchNorm2d(1024//d_size),
            nn.ELU(),
            nn.ConvTranspose2d(in_channels=1024//d_size, out_channels=512//d_size, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(512//d_size),
            nn.ELU(),
            nn.ConvTranspose2d(in_channels=512//d_size, out_channels=256//d_size, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(256//d_size),
            nn.ELU(),
            nn.ConvTranspose2d(in_channels=256//d_size, out_channels=128//d_size, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128//d_size),
            nn.ELU(),
        )


        # if args.less_upsample is False:
        # self.p_x_layers = nn.Sequential(self.p_x_layers,)


        if self.args.input_type == 'gray' or self.args.input_type == 'continuous':
            if args.use_logit is False:
                self.p_x_mean = nn.Sequential(nn.Conv2d(in_channels=128//d_size, out_channels=3, kernel_size=5, stride=1, padding=2), nn.Sigmoid())
            else:
                self.p_x_mean = nn.Sequential(nn.Conv2d(in_channels=128//d_size, out_channels=3, kernel_size=5, stride=1, padding=2))
