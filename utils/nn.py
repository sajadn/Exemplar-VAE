import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def xavier_init(m):
    s =  np.sqrt( 2. / (m.in_features + m.out_features) )
    m.weight.data.normal_(0, s)


def he_init(m):
    s =  np.sqrt( 2. / m.in_features )
    m.weight.data.normal_(0, s)


def normal_init(m, mean=0., std=0.01):
    m.weight.data.normal_(mean, std)


class CReLU(nn.Module):
    def __init__(self):
        super(CReLU, self).__init__()

    def forward(self, x):
        return torch.cat( F.relu(x), F.relu(-x), 1 )


class NonLinear(nn.Module):
    def __init__(self, input_size, output_size, bias=True, activation=None):
        super(NonLinear, self).__init__()

        self.activation = activation
        self.linear = nn.Linear(int(input_size), int(output_size), bias=bias)

    def forward(self, x):
        h = self.linear(x)
        if self.activation is not None:
            h = self.activation( h )

        return h


class GatedDense(nn.Module):
    def __init__(self, input_size, output_size, activation=None, no_attention=False):
        super(GatedDense, self).__init__()

        self.activation = activation
        self.no_attention = no_attention
        self.sigmoid = nn.Sigmoid()
        self.h = nn.Linear(input_size, output_size)
        if no_attention is False:
            self.g = nn.Linear(input_size, output_size)
        else:
            self.activation = torch.nn.ReLU()

    def forward(self, x):
        h = self.h(x)
        if self.activation is not None:
            h = self.activation( self.h( x ) )
        try:
            if self.no_attention is False:
                g = self.sigmoid(self.g(x))
                return h * g
            else:
                return h
        except:
            g = self.sigmoid(self.g(x))
            return h * g


class GatedConv2d(nn.Module):
    def __init__(self, input_channels, output_channels, kernel_size, stride, padding, dilation=1, activation=None,
                 no_attention=False):
        super(GatedConv2d, self).__init__()
        self.no_attention = no_attention

        self.activation = activation
        self.sigmoid = nn.Sigmoid()

        self.h = nn.Conv2d(input_channels, output_channels, kernel_size, stride, padding, dilation)
        if no_attention is False:
            self.g = nn.Conv2d(input_channels, output_channels, kernel_size, stride, padding, dilation)
        else:
            self.activation = torch.nn.ELU()

    def forward(self, x):
        if self.activation is None:
            h = self.h(x)
        else:
            h = self.activation( self.h( x ) )

        # if self.no_attention is False:
        g = self.sigmoid( self.g( x ) )
        return h * g
        # else:
        #     return h


class Conv2d(nn.Module):
    def __init__(self, input_channels, output_channels, kernel_size, stride, padding, dilation=1, activation=None, bias=True):
        super(Conv2d, self).__init__()

        self.activation = activation
        self.conv = nn.Conv2d(input_channels, output_channels, kernel_size, stride, padding, dilation, bias=bias)

    def forward(self, x):
        h = self.conv(x)
        if self.activation is None:
            out = h
        else:
            out = self.activation(h)

        return out






