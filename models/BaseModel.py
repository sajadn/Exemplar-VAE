from __future__ import print_function
import numpy as np
import torch
import torch.utils.data
import torch.nn as nn
from torch.autograd import Variable
from utils.nn import normal_init, NonLinear
from utils.distributions import log_normal_diag_vectorized
import math
from utils.nn import he_init
from utils.distributions import pairwise_distance
from utils.distributions import log_bernoulli, log_normal_diag, log_normal_standard, log_logistic_256
from abc import ABC, abstractmethod
from torch import _weight_norm
from utils.utils import reparameterize


class BaseModel(nn.Module, ABC):
    def __init__(self, args):
        super(BaseModel, self).__init__()
        print("constructor")
        self.args = args

        if self.args.prior == 'vampprior':
            self.add_pseudoinputs()

        if self.args.prior == 'exemplar_prior':
            self.prior_log_variance = torch.nn.Parameter(torch.randn((1)))

        if self.args.input_type == 'binary':
            self.p_x_mean = NonLinear(self.args.hidden_size, np.prod(self.args.input_size), activation=nn.Sigmoid())
        elif self.args.input_type == 'gray' or self.args.input_type == 'continuous':
            self.p_x_mean = NonLinear(self.args.hidden_size, np.prod(self.args.input_size))
            self.p_x_logvar = NonLinear(self.args.hidden_size, np.prod(self.args.input_size),
                                        activation=nn.Hardtanh(min_val=-4.5, max_val=0))
            self.decoder_logstd = torch.nn.Parameter(torch.randn(self.args.input_size[0], ), requires_grad=True)

        self.create_model(args)
        self.he_initializer()

    def he_initializer(self):
        print("he initializer")

        for m in self.modules():
            if isinstance(m, nn.Linear):
                he_init(m)

    @abstractmethod
    def create_model(self, args):
        pass

    def kl_loss(self, latent_stats, exemplars_embedding, dataset, cache, x_indices):
        kl = 0
        for i in range(len(latent_stats)-1):
            z_q, z_q_mean, z_q_logvar, z_p_mean, z_p_logvar = latent_stats[i]
            log_p_z = log_normal_diag(z_q.view(-1, self.args.z1_size),
                                       z_p_mean.view(-1, self.args.z1_size),
                                       z_p_logvar.view(-1, self.args.z1_size), dim=1)
            log_q_z = log_normal_diag(z_q.view(-1, self.args.z1_size),
                                      z_q_mean.view(-1, self.args.z1_size),
                                      z_q_logvar.view(-1, self.args.z1_size), dim=1)
            kl += -(log_p_z-log_q_z)
        z_q, z_q_mean, z_q_logvar = latent_stats[-1]

        if exemplars_embedding is None and self.args.prior == 'exemplar_prior':
            exemplars_embedding = self.get_exemplar_set(z_q_mean, z_q_logvar, dataset, cache, x_indices)

        log_p_z = self.log_p_z(z=(z_q, x_indices), exemplars_embedding=exemplars_embedding)
        log_q_z = log_normal_diag(z_q.view(-1, self.args.z2_size),
                                   z_q_mean.view(-1, self.args.z2_size),
                                   z_q_logvar.view(-1, self.args.z2_size), dim=1)
        kl += -(log_p_z - log_q_z)
        return kl



    def reconstruction_loss(self, x, x_mean, x_logvar):
        if self.args.input_type == 'binary':
            return log_bernoulli(x, x_mean, dim=1)
        elif self.args.input_type == 'gray' or self.args.input_type == 'continuous':
            if self.args.use_logit is True:
                return log_normal_diag(x, x_mean, x_logvar, dim=1)
            else:
                return log_logistic_256(x, x_mean, x_logvar, dim=1)
        else:
            raise Exception('Wrong input type!')

    def calculate_loss(self, x, beta=1., average=False,
                       exemplars_embedding=None, cache=None, dataset=None):
        x, x_indices = x
        x_mean, x_logvar, latent_stats = self.forward(x)
        RE = self.reconstruction_loss(x, x_mean, x_logvar)
        KL = self.kl_loss(latent_stats, exemplars_embedding, dataset, cache, x_indices)
        loss = -RE + beta*KL
        if average:
            loss = torch.mean(loss)
            RE = torch.mean(RE)
            KL = torch.mean(KL)

        return loss, RE, KL

    def log_p_z_vampprior(self, z, exemplars_embedding):
        if exemplars_embedding is None:
            C = self.args.number_components
            X = self.means(self.idle_input)
            z_p_mean, z_p_logvar = self.q_z(X, prior=True)  # C x M
        else:
            C = torch.tensor(self.args.number_components).float()
            z_p_mean, z_p_logvar = exemplars_embedding

        z_expand = z.unsqueeze(1)
        means = z_p_mean.unsqueeze(0)
        logvars = z_p_logvar.unsqueeze(0)
        return log_normal_diag(z_expand, means, logvars, dim=2) - math.log(C)

    def log_p_z_exemplar(self, z, z_indices, exemplars_embedding, test):
        centers, center_log_variance, center_indices = exemplars_embedding
        denominator = torch.tensor(len(centers)).expand(len(z)).float().to(self.args.device)
        center_log_variance = center_log_variance[0, :].unsqueeze(0)
        prob, _ = log_normal_diag_vectorized(z, centers, center_log_variance)  # MB x C
        if test is False and self.args.no_mask is False:
            mask = z_indices.expand(-1, len(center_indices)) \
                    == center_indices.squeeze().unsqueeze(0).expand(len(z_indices), -1)
            prob.masked_fill_(mask, value=float('-inf'))
            denominator = denominator - mask.sum(dim=1).float()
        prob -= torch.log(denominator).unsqueeze(1)
        return prob

    def log_p_z(self, z, exemplars_embedding, sum=True, test=None):
        z, z_indices = z
        if test is None:
            test = not self.training
        if self.args.prior == 'standard':
            return log_normal_standard(z, dim=1)
        elif self.args.prior == 'vampprior':
            prob = self.log_p_z_vampprior(z, exemplars_embedding)
        elif self.args.prior == 'exemplar_prior':
            prob = self.log_p_z_exemplar(z, z_indices, exemplars_embedding, test)
        else:
            raise Exception('Wrong name of the prior!')
        if sum:
            prob_max, _ = torch.max(prob, 1)  # MB x 1
            log_prior = prob_max + torch.log(torch.sum(torch.exp(prob - prob_max.unsqueeze(1)), 1))  # MB x 1
        else:
            return prob
        return log_prior

    def add_pseudoinputs(self):
        nonlinearity = nn.Hardtanh(min_val=0.0, max_val=1.0)
        self.means = NonLinear(self.args.number_components, np.prod(self.args.input_size), bias=False, activation=nonlinearity)
        # init pseudo-inputs
        if self.args.use_training_data_init:
            self.means.linear.weight.data = self.args.pseudoinputs_mean
        else:
            normal_init(self.means.linear, self.args.pseudoinputs_mean, self.args.pseudoinputs_std)
        self.idle_input = Variable(torch.eye(self.args.number_components, self.args.number_components), requires_grad=False)
        self.idle_input = self.idle_input.to(self.args.device)

    def generate_z_interpolate(self, exemplars_embedding=None, dim=0):
        new_zs = []
        exemplars_embedding, _, _ = exemplars_embedding
        step_counts = 10
        step = (exemplars_embedding[1] - exemplars_embedding[0])/step_counts
        for i in range(step_counts):
            new_z = exemplars_embedding[0].clone()
            new_z += i*step
            new_zs.append(new_z.unsqueeze(0))
        return torch.cat(new_zs, dim=0)

    def generate_z(self, N=25, dataset=None):
        if self.args.prior == 'standard':
            z_sample_rand = torch.FloatTensor(N, self.args.z1_size).normal_().to(self.args.device)
        elif self.args.prior == 'vampprior':
            means = self.means(self.idle_input)[0:N]
            z_sample_gen_mean, z_sample_gen_logvar = self.q_z(means)
            z_sample_rand = reparameterize(z_sample_gen_mean, z_sample_gen_logvar)
            z_sample_rand = z_sample_rand.to(self.args.device)
        elif self.args.prior == 'exemplar_prior':
            rand_indices, _ = torch.sort(torch.randint(low=0, high=self.args.training_set_size, size=(N,)))
            exemplars = dataset[rand_indices.detach().cpu().numpy()][0]
            z_sample_gen_mean, z_sample_gen_logvar = self.q_z(exemplars.to(self.args.device), prior=True)
            z_sample_rand = reparameterize(z_sample_gen_mean, z_sample_gen_logvar)
            z_sample_rand = z_sample_rand.to(self.args.device)
        return z_sample_rand

    def reference_based_generation_z(self, N=25, reference_image=None):
        pseudo, log_var = self.q_z(reference_image.to(self.args.device), prior=True)
        pseudo = pseudo.unsqueeze(1).expand(-1, N, -1).reshape(-1, pseudo.shape[-1])
        log_var = log_var[0].unsqueeze(0).expand(len(pseudo), -1)
        z_sample_rand = reparameterize(pseudo, log_var)
        z_sample_rand = z_sample_rand.reshape(-1, N, pseudo.shape[1])
        return z_sample_rand

    def reconstruct_x(self, x):
        x_reconstructed, _, _ = self.forward(x)
        return x_reconstructed

    def generate_x(self, N=25, dataset=None):
        z2_sample_rand = self.generate_z(N=N, dataset=dataset)
        return self.generate_x_from_z(z2_sample_rand)

    def reference_based_generation_x(self, N=25, reference_image=None):
        z2_sample_rand = \
            self.reference_based_generation_z(N=N, reference_image=reference_image)
        generated_x = self.generate_x_from_z(z2_sample_rand.reshape(-1, self.args.z2_size))
        return generated_x

    def generate_x_interpolate(self, exemplars_embedding, dim=0):
        zs = self.generate_z_interpolate(exemplars_embedding, dim=dim)
        print(zs.shape)
        return self.generate_x_from_z(zs, with_reparameterize=False)

    def reshape_variance(self, variance, shape):
        return variance[0]*torch.ones(shape).to(self.args.device)

    def q_z(self, x, z1=None, prior=False):
        if 'conv' in self.args.model_name or self.args.model_name=='CelebA':
            x = x.view(-1, *self.args.input_size)
        h = self.q_z_layers(x)
        if self.args.model_name == 'convhvae_2level':
            h = h.view(x.size(0), -1)
        z_q_mean = self.q_z_mean(h)
        if prior is True:
            if self.args.prior == 'exemplar_prior':
                z_q_logvar = self.prior_log_variance * torch.ones((x.shape[0], self.args.z1_size)).to(self.args.device)
            else:
                z_q_logvar = self.q_z_logvar(h)
        else:
            z_q_logvar = self.q_z_logvar(h)
        return z_q_mean.reshape(-1, self.args.z1_size), z_q_logvar.reshape(-1, self.args.z1_size)

    def cache_z(self, dataset, prior=True, cuda=True):
        cached_z = []
        cached_log_var = []
        caching_batch_size = 5000
        num_batchs = math.ceil(len(dataset) / caching_batch_size)
        for i in range(num_batchs):
            if len(dataset[0]) == 3:
                batch_data, batch_indices, _ = dataset[i * caching_batch_size:(i + 1) * caching_batch_size]
            else:
                batch_data, _ = dataset[i * caching_batch_size:(i + 1) * caching_batch_size]

            exemplars_embedding, log_variance_z = self.q_z(batch_data.to(self.args.device), prior=prior)
            cached_z.append(exemplars_embedding)
            cached_log_var.append(log_variance_z)
        cached_z = torch.cat(cached_z, dim=0)
        cached_log_var = torch.cat(cached_log_var, dim=0)
        cached_z = cached_z.to(self.args.device)
        cached_log_var = cached_log_var.to(self.args.device)
        return cached_z, cached_log_var

    def get_exemplar_set(self, z_mean, z_log_var, dataset, cache, x_indices):
        if self.args.approximate_prior is False:
            exemplars_indices = torch.randperm(self.args.training_set_size)[:self.args.number_components]
            exemplars_z, log_variance = self.q_z(dataset[exemplars_indices][0].to(self.args.device), prior=True)
            exemplar_set = (exemplars_z, log_variance, exemplars_indices.to(self.args.device))
        else:
            exemplar_set = self.get_approximate_nearest_exemplars(
                z=(z_mean, z_log_var, x_indices),
                dataset=dataset,
                cache=cache)
        return exemplar_set

    def get_approximate_nearest_exemplars(self, z, cache, dataset):
        exemplars_indices = torch.randperm(self.args.training_set_size)[:self.args.number_components].to(self.args.device)
        z, _, indices = z
        cached_z, cached_log_variance = cache
        cached_z[indices.reshape(-1)] = z
        sub_cache = cached_z[exemplars_indices, :]
        _, nearest_indices = pairwise_distance(z, sub_cache) \
            .topk(k=self.args.approximate_k, largest=False, dim=1)
        nearest_indices = torch.unique(nearest_indices.view(-1))
        exemplars_indices, _ = torch.sort(exemplars_indices[nearest_indices].view(-1))
        exemplars = dataset[exemplars_indices.detach().cpu().numpy()][0].to(self.args.device)
        exemplars_z, log_variance = self.q_z(exemplars, prior=True)
        cached_z[exemplars_indices] = exemplars_z
        exemplar_set = (exemplars_z, log_variance, exemplars_indices)
        return exemplar_set

    def data_dependent_init(self, data_batch):
        def init_hook_(module, input, output):
            g = getattr(module, 'weight_g')
            g.data = g.new_ones(g.shape)
            setattr(module, 'weight', _weight_norm(getattr(module, 'weight_v'), g, dim=0))
            out_v = module.conv2d_forward(input[0], module.weight)
            std, mean = torch.std_mean(out_v, dim=[0, 2, 3])
            g.data = (.1*g.data)/std.reshape((len(std), 1, 1, 1))
            b = getattr(module, 'bias')
            b.data = (b.data-mean)*g.data.squeeze()
            setattr(module, 'weight', _weight_norm(getattr(module, 'weight_v'), g, dim=0))
            return module.conv2d_forward(input[0], module.weight)

        handles = []
        for m in self.modules():
            if isinstance(m, torch.nn.Conv2d):
                handles.append(m.register_forward_hook(init_hook_))

        self.forward(data_batch)
        for h in handles:
            h.remove()
