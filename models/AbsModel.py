from __future__ import print_function
import numpy as np
import torch
import torch.utils.data
from models.BaseModel import BaseModel
from utils.distributions import log_normal_diag
from utils.utils import inverse_scaled_logit
from utils.plot_images import imshow

class AbsModel(BaseModel):
    def __init__(self, args):
        super(AbsModel, self).__init__(args)

    def kl_loss(self, latent_stats, exemplars_embedding, dataset, cache, x_indices):
        z_q, z_q_mean, z_q_logvar = latent_stats
        if exemplars_embedding is None and self.args.prior == 'exemplar_prior':
            exemplars_embedding = self.get_exemplar_set(z_q_mean, z_q_logvar, dataset, cache, x_indices)
        log_p_z = self.log_p_z(z=(z_q, x_indices), exemplars_embedding=exemplars_embedding)
        log_q_z = log_normal_diag(z_q, z_q_mean, z_q_logvar, dim=1)
        return -(log_p_z - log_q_z)

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
        if self.args.model_name == 'CelebA':
            z = z.reshape(-1, args.z1_size//16, 4, 4)
        z = self.p_x_layers(z)
        x_mean = self.p_x_mean(z)
        if self.args.input_type == 'binary':
            x_logvar = torch.zeros(1, np.prod(self.args.input_size))
        else:
            if self.args.use_logit is False:
                if self.args.zero_center:
                    x_mean += 0.5
                # x_mean = torch.clamp(x_mean, min=0.+1./512., max=1.-1./512.)
            try:
                decoder_logstd = torch.clamp(self.decoder_logstd, max=self.args.decoder_upper_bound)
            except:
                decoder_logstd = self.decoder_logstd
            reshaped_var = decoder_logstd.unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
            x_logvar = reshaped_var*x_mean.new_ones(size=x_mean.shape)
        return x_mean.reshape(-1, np.prod(self.args.input_size)),\
               x_logvar.reshape(-1, np.prod(self.args.input_size))

    def forward(self, x, label=0, num_categories=10):
        z_q_mean, z_q_logvar = self.q_z(x)

        z_q = self.reparameterize(z_q_mean, z_q_logvar)
        x_mean, x_logvar = self.p_x(z_q)

        return x_mean, x_logvar, (z_q, z_q_mean, z_q_logvar)
