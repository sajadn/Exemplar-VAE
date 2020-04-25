from __future__ import print_function
import numpy as np
import torch
import torch.utils.data
from models.BaseModel import BaseModel
from utils.distributions import log_normal_diag
from utils.utils import inverse_scaled_logit
from utils.plot_images import imshow
from utils.utils import reparameterize

class AbsModel(BaseModel):
    def __init__(self, args):
        super(AbsModel, self).__init__(args)

    def generate_x_from_z(self, z, with_reparameterize=True):
        generated_x, _ = self.p_x(z)
        try:
            if self.args.use_logit is True:
                return torch.floor(inverse_scaled_logit(generated_x, self.args.lambd)*256).int()
            else:
                return generated_x
        except Exception as e:
            print(e)
            return generated_x

    def p_x(self, z):
        if 'conv' in self.args.model_name:
            z = z.reshape(-1, self.bottleneck, self.args.input_size[1]//2, self.args.input_size[1]//2)
        z = self.p_x_layers(z)
        x_mean = self.p_x_mean(z)
        if self.args.input_type == 'binary':
            x_logvar = torch.zeros(1, np.prod(self.args.input_size))
        else:
            if self.args.use_logit is False:
                x_mean = torch.clamp(x_mean, min=0.+1./512., max=1.-1./512.)
            reshaped_var = self.decoder_logstd.unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
            x_logvar = reshaped_var*x_mean.new_ones(size=x_mean.shape)
        return x_mean.reshape(-1, np.prod(self.args.input_size)),\
               x_logvar.reshape(-1, np.prod(self.args.input_size))

    def forward(self, x):
        z_q_mean, z_q_logvar = self.q_z(x)
        z_q = reparameterize(z_q_mean, z_q_logvar)
        x_mean, x_logvar = self.p_x(z_q)
        return x_mean, x_logvar, [(z_q, z_q_mean, z_q_logvar)]
