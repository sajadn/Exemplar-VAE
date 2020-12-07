from __future__ import print_function
from utils.plot_images import plot_images
import torch
import time
from scipy.special import logsumexp
import numpy as np
from utils.utils import load_model
import torch.nn.functional as F


def evaluate_loss(args, model, loader, dataset=None, exemplars_embedding=None):
    evaluateed_elbo, evaluate_re, evaluate_kl = 0, 0, 0
    model.eval()
    if exemplars_embedding is None:
        exemplars_embedding = load_all_pseudo_input(args, model, dataset)

    for data in loader:
        if len(data) == 3:
            data, _, _ = data
        else:
            data, _ = data
        data = data.to(args.device)
        x = data
        x_indices = None
        x = (x, x_indices)
        loss, RE, KL = model.calculate_loss(x, average=False, exemplars_embedding=exemplars_embedding)
        evaluateed_elbo += loss.sum().item()
        evaluate_re += -RE.sum().item()
        evaluate_kl += KL.sum().item()
    evaluateed_elbo /= len(loader.dataset)
    evaluate_re /= len(loader.dataset)
    evaluate_kl /= len(loader.dataset)
    return evaluateed_elbo, evaluate_re, evaluate_kl


def visualize_reconstruction(test_samples, model, args, dir):
    samples_reconstruction = model.reconstruct_x(test_samples[0:25])

    if args.use_logit:
        test_samples = model.logit_inverse(test_samples)
        samples_reconstruction = model.logit_inverse(samples_reconstruction)
    plot_images(args, test_samples.cpu().numpy()[0:25], dir, 'real', size_x=5, size_y=5)
    plot_images(args, samples_reconstruction.cpu().numpy(), dir, 'reconstructions', size_x=5, size_y=5)


def visualize_generation(dataset, model, args, dir):
    generation_rounds = 1
    for i in range(generation_rounds):
        samples_rand = model.generate_x(25, dataset=dataset)
        plot_images(args, samples_rand.cpu().numpy(), dir, 'generations_{}'.format(i), size_x=5, size_y=5)
    if args.prior == 'vampprior':
        pseudo_means = model.means(model.idle_input)
        plot_images(args, pseudo_means[0:25].cpu().numpy(), dir, 'pseudoinputs', size_x=5, size_y=5)


def load_all_pseudo_input(args, model, dataset):
    if args.prior == 'exemplar_prior':
        exemplars_z, exemplars_log_var = model.cache_z(dataset)
        embedding = (exemplars_z, exemplars_log_var, torch.arange(len(exemplars_z)))
    elif args.prior == 'vampprior':
        pseudo_means = model.means(model.idle_input)
        if 'conv' in args.model_name:
            pseudo_means = pseudo_means.view(-1, args.input_size[0], args.input_size[1], args.input_size[2])
        embedding = model.q_z(pseudo_means, prior=True)  # C x M
    elif args.prior == 'standard':
        embedding = None
    else:
        raise Exception("wrong name of prior")
    return embedding


def calculate_likelihood(args, model, loader, S=5000, exemplars_embedding=None):
    likelihood_test = []
    batch_size_evaluation = 1
    auxilary_loader = torch.utils.data.DataLoader(loader.dataset, batch_size=batch_size_evaluation)
    t0 = time.time()
    for index, (data, _) in enumerate(auxilary_loader):
        data = data.to(args.device)
        if index % 100 == 0:
            print(time.time() - t0)
            t0 = time.time()
            print('{:.2f}%'.format(index / (1. * len(auxilary_loader)) * 100))
        x = data.expand(S, data.size(1))
        if args.model_name == 'pixelcnn':
            BS = S//100
            prob = []
            for i in range(BS):
                bx = x[i*100:(i+1)*100]
                x_indices = None
                bprob, _, _ = model.calculate_loss((bx, x_indices), exemplars_embedding=exemplars_embedding)
                prob.append(bprob)
            prob = torch.cat(prob, dim=0)
        else:
            x_indices = None
            prob, _, _ = model.calculate_loss((x, x_indices), exemplars_embedding=exemplars_embedding)
        likelihood_x = logsumexp(-prob.cpu().numpy())
        if model.args.use_logit:
            lambd = torch.tensor(model.args.lambd).float()
            likelihood_x -= (-F.softplus(-x) - F.softplus(x)\
                             - torch.log((1 - 2 * lambd)/256)).sum(dim=1).cpu().numpy()
        likelihood_test.append(likelihood_x - np.log(len(prob)))
    likelihood_test = np.array(likelihood_test)
    return -np.mean(likelihood_test)


def final_evaluation(train_loader, test_loader, valid_loader, best_model_path_load,
                     model, optimizer, args, dir):
        _ = load_model(best_model_path_load, model, optimizer)
        model.eval()
        exemplars_embedding = load_all_pseudo_input(args, model, train_loader.dataset)
        test_samples = next(iter(test_loader))[0].to(args.device)
        visualize_reconstruction(test_samples, model, args, dir)
        visualize_generation(train_loader.dataset, model, args, dir)
        test_elbo, test_re, test_kl = evaluate_loss(args, model, test_loader, dataset=train_loader.dataset, exemplars_embedding=exemplars_embedding)
        valid_elbo, valid_re, valid_kl = evaluate_loss(args, model, valid_loader, dataset=valid_loader.dataset, exemplars_embedding=exemplars_embedding)
        train_elbo, _, _ = evaluate_loss(args, model, train_loader, dataset=train_loader.dataset, exemplars_embedding=exemplars_embedding)
        test_log_likelihood = calculate_likelihood(args, model, test_loader, exemplars_embedding=exemplars_embedding, S=args.S)
        final_evaluation_txt = 'FINAL EVALUATION ON TEST SET\n' \
                               'LogL (TEST): {:.2f}\n' \
                               'LogL (TRAIN): {:.2f}\n' \
                               'ELBO (TEST): {:.2f}\n' \
                               'ELBO (TRAIN): {:.2f}\n' \
                               'ELBO (VALID): {:.2f}\n' \
                               'RE: {:.2f}\n' \
                               'KL: {:.2f}'.format(
            test_log_likelihood,
            0,
            test_elbo,
            train_elbo,
            valid_elbo,
            test_re,
            test_kl)

        print(final_evaluation_txt)
        with open(dir + 'vae_experiment_log.txt', 'a') as f:
            print(final_evaluation_txt, file=f)
        torch.save(test_log_likelihood, dir + args.model_name + '.test_log_likelihood')
        torch.save(test_elbo, dir + args.model_name + '.test_loss')
        torch.save(test_re, dir + args.model_name + '.test_re')
        torch.save(test_kl, dir + args.model_name + '.test_kl')


# TODO remove last loop from this function
def compute_mean_variance_per_dimension(args, model, test_loader):
    means = []
    for batch, _ in test_loader:
        mean, _ = model.q_z(batch.to(args.device))
        means.append(mean)
    means = torch.cat(means, dim=0).cpu().detach().numpy()
    active = 0
    for i in range(means.shape[1]):
        if np.var(means[:, i].reshape(-1)) > 0.01:
            active += 1
    print('active dimensions', active)
    return active


