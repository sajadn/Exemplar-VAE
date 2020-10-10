from __future__ import print_function
import argparse
import datetime
from utils.load_data.data_loader_instances import load_dataset
from utils.utils import importing_model
import torch
import math
import os
from utils.utils import save_model, load_model
from utils.optimizer import AdamNormGrad
import time
from utils.training import train_one_epoch
from utils.evaluation import evaluate_loss, final_evaluation
import random

def str2bool(v):
    if isinstance(v, bool):
       return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


parser = argparse.ArgumentParser(description='VAE+VampPrior')
parser.add_argument('--batch_size', type=int, default=100, metavar='BStrain',
                    help='input batch size for training (default: 100)')
parser.add_argument('--test_batch_size', type=int, default=100, metavar='BStest',
                    help='input batch size for testing (default: 100)')
parser.add_argument('--epochs', type=int, default=2000, metavar='E',
                    help='number of epochs to train (default: 2000)')
parser.add_argument('--lr', type=float, default=0.0005, metavar='LR',
                    help='learning rate (default: 0.0005)')
parser.add_argument('--early_stopping_epochs', type=int, default=50, metavar='ES',
                    help='number of epochs for early stopping')
parser.add_argument('--z1_size', type=int, default=40, metavar='M1',
                    help='latent size')
parser.add_argument('--z2_size', type=int, default=40, metavar='M2',
                    help='latent size')
parser.add_argument('--input_size', type=int, default=[1, 28, 28], metavar='D',
                    help='input size')
parser.add_argument('--number_components', type=int, default=50000, metavar='NC',
                    help='number of pseudo-inputs')
parser.add_argument('--pseudoinputs_mean', type=float, default=-0.05, metavar='PM',
                    help='mean for init pseudo-inputs')
parser.add_argument('--pseudoinputs_std', type=float, default=0.01, metavar='PS',
                    help='std for init pseudo-inputs')
parser.add_argument('--use_training_data_init', action='store_true', default=False,
                    help='initialize pseudo-inputs with randomly chosen training data')
parser.add_argument('--model_name', type=str, default='vae', metavar='MN',
                    help='model name: vae, hvae_2level, convhvae_2level')
parser.add_argument('--prior', type=str, default='vampprior', metavar='P',
                    help='prior: standard, vampprior, exemplar_prior')
parser.add_argument('--input_type', type=str, default='binary', metavar='IT',
                    help='type of the input: binary, gray, continuous, pca')
parser.add_argument('--S', type=int, default=5000, metavar='SLL',
                    help='number of samples used for approximating log-likelihood,'
                         'i.e. number of samples in IWAE')
parser.add_argument('--MB', type=int, default=100, metavar='MBLL',
                    help='size of a mini-batch used for approximating log-likelihood')
parser.add_argument('--use_whole_train', type=str2bool, default=False,
                    help='use whole training data points at the test time')
parser.add_argument('--dataset_name', type=str, default='freyfaces', metavar='DN',
                    help='name of the dataset: static_mnist, dynamic_mnist, omniglot, caltech101silhouettes,'
                         ' histopathologyGray, freyfaces, cifar10')
parser.add_argument('--dynamic_binarization', action='store_true', default=False,
                    help='allow dynamic binarization')
parser.add_argument('--seed', type=int, default=14, metavar='S',
                    help='random seed (default: 14)')

parser.add_argument('--no_mask', action='store_true', default=False, help='no leave one out')

parser.add_argument('--parent_dir', type=str, default='')
parser.add_argument('--same_variational_var', type=str2bool, default=False,
                    help='use same variance for different dimentions')
parser.add_argument('--model_signature', type=str, default='', help='load from this directory and continue training')
parser.add_argument('--warmup', type=int, default=100, metavar='WU',
                    help='number of epochs for warmu-up')
parser.add_argument('--slurm_task_id', type=str, default='')
parser.add_argument('--slurm_job_id', type=str, default='')
parser.add_argument('--approximate_prior', type=str2bool, default=False)
parser.add_argument('--just_evaluate', type=str2bool, default=False)
parser.add_argument('--no_attention', type=str2bool, default=False)
parser.add_argument('--approximate_k', type=int, default=10)
parser.add_argument('--hidden_size', type=int, default=300)
parser.add_argument('--base_dir', type=str, default='snapshots/')
parser.add_argument('--continuous', type=str2bool, default=False)
parser.add_argument('--use_logit', type=str2bool, default=False)
parser.add_argument('--lambd', type=float, default=1e-4)
parser.add_argument('--bottleneck', type=int, default=6)
parser.add_argument('--training_set_size', type=int, default=50000)


def initial_or_load(checkpoint_path_load, model, optimizer, dir):
    if os.path.exists(checkpoint_path_load):
        model_loaded_str = "******model is loaded*********"
        print(model_loaded_str)
        with open(dir + 'whole_log.txt', 'a') as f:
            print(model_loaded_str, file=f)
        checkpoint = load_model(checkpoint_path_load, model, optimizer)
        begin_epoch = checkpoint['epoch']
        best_loss = checkpoint['best_loss']
        e = checkpoint['e']
    else:
        begin_epoch = 1
        best_loss = math.inf
        e = 0
    return begin_epoch, best_loss, e


def save_loss_files(folder, train_loss_history,
                    train_re_history, train_kl_history, val_loss_history, val_re_history, val_kl_history):
    torch.save(train_loss_history, folder + '.train_loss')
    torch.save(train_re_history, folder + '.train_re')
    torch.save(train_kl_history, folder + '.train_kl')
    torch.save(val_loss_history, folder + '.val_loss')
    torch.save(val_re_history, folder + '.val_re')
    torch.save(val_kl_history, folder + '.val_kl')


def run_density_estimation(args, train_loader_input, val_loader_input, test_loader_input, model, optimizer, dir, model_name='vae'):
    torch.save(args, dir + args.model_name + '.config')
    train_loss_history, train_re_history, train_kl_history, val_loss_history, val_re_history, val_kl_history, \
    time_history = [], [], [], [], [], [], []
    checkpoint_path_save = os.path.join(dir, 'checkpoint_temp.pth')
    checkpoint_path_load = os.path.join(dir, 'checkpoint.pth')
    best_model_path_load = os.path.join(dir, 'checkpoint_best.pth')
    decayed = False
    time_history = []
    # with torch.autograd.detect_anomaly():
    begin_epoch, best_loss, e = initial_or_load(checkpoint_path_load, model, optimizer, dir)
    if args.just_evaluate is False:
        for epoch in range(begin_epoch, args.epochs + 1):
            time_start = time.time()
            train_loss_epoch, train_re_epoch, train_kl_epoch \
                = train_one_epoch(epoch, args, train_loader_input, model, optimizer)
            with torch.no_grad():
                val_loss_epoch, val_re_epoch, val_kl_epoch = evaluate_loss(args, model, val_loader_input,
                                                                           dataset=train_loader_input.dataset)
            time_end = time.time()
            time_elapsed = time_end - time_start
            content = {'epoch': epoch, 'state_dict': model.state_dict(),
                       'optimizer': optimizer.state_dict(), 'best_loss': best_loss, 'e': e}
            if epoch % 10 == 0:
                save_model(checkpoint_path_save, checkpoint_path_load, content)
            if val_loss_epoch < best_loss:
                e = 0
                best_loss = val_loss_epoch
                print('->model saved<-')
                save_model(checkpoint_path_save, best_model_path_load, content)
            else:
                e += 1
                if epoch < args.warmup:
                    e = 0
                if e > args.early_stopping_epochs:
                    break

            if math.isnan(val_loss_epoch):
                print("***** val loss is Nan *******")
                break

            for param_group in optimizer.param_groups:
                learning_rate = param_group['lr']
                break

            time_history.append(time_elapsed)

            epoch_report = 'Epoch: {}/{}, Time elapsed: {:.2f}s\n' \
                           'learning rate: {:.5f}\n' \
                           '* Train loss: {:.2f}   (RE: {:.2f}, KL: {:.2f})\n' \
                           'o Val.  loss: {:.2f}   (RE: {:.2f}, KL: {:.2f})\n' \
                           '--> Early stopping: {}/{} (BEST: {:.2f})\n'.format(epoch, args.epochs, time_elapsed,
                                                                               learning_rate,
                                                                               train_loss_epoch, train_re_epoch,
                                                                               train_kl_epoch, val_loss_epoch,
                                                                               val_re_epoch, val_kl_epoch, e,
                                                                               args.early_stopping_epochs, best_loss)

            if args.prior == 'exemplar_prior':
                print("Prior Variance", model.prior_log_variance.item())
            if args.continuous is True:
                print("Decoder Variance", model.decoder_logstd.item())
            print(epoch_report)
            with open(dir + 'whole_log.txt', 'a') as f:
                print(epoch_report, file=f)

            train_loss_history.append(train_loss_epoch), train_re_history.append(
                train_re_epoch), train_kl_history.append(train_kl_epoch)
            val_loss_history.append(val_loss_epoch), val_re_history.append(val_re_epoch), val_kl_history.append(
                val_kl_epoch)

        save_loss_files(dir + args.model_name, train_loss_history,
                        train_re_history, train_kl_history, val_loss_history, val_re_history, val_kl_history)

    with torch.no_grad():
        final_evaluation(train_loader_input, test_loader_input, val_loader_input,
                         best_model_path_load, model, optimizer, args, dir)


def run(args, kwargs):
    print('create model')
    # importing model
    VAE = importing_model(args)
    print('load data')
    train_loader, val_loader, test_loader, args = load_dataset(args, use_fixed_validation=True, **kwargs)
    if args.slurm_job_id != '':
        args.model_signature = str(args.seed)
        # base_dir = 'checkpoints/final_report/'
    elif args.model_signature == '':
        args.model_signature = str(datetime.datetime.now())[0:19]

    if args.parent_dir == '':
        args.parent_dir = args.prior + '_on_' + args.dataset_name+'_model_name='+args.model_name
    model_name = args.dataset_name + '_' + args.model_name + '_' + args.prior \
                  + '_(components_' + str(args.number_components) + ', lr=' + str(args.lr) + ')'
    snapshots_path = os.path.join(args.base_dir, args.parent_dir) + '/'
    dir = snapshots_path + args.model_signature + '_' + model_name + '_' + args.parent_dir + '/'

    if args.just_evaluate:
        config = torch.load(dir + args.model_name + '.config')
        config.translation = False
        config.hidden_size = 300
        model = VAE(config)
    else:
        model = VAE(args)
    if not os.path.exists(dir):
        os.makedirs(dir)
    model.to(args.device)
    optimizer = AdamNormGrad(model.parameters(), lr=args.lr)
    print(args)
    config_file = dir+'vae_config.txt'
    with open(config_file, 'a') as f:
        print(args, file=f)
    run_density_estimation(args, train_loader, val_loader, test_loader, model, optimizer, dir, model_name = args.model_name)


if __name__ == "__main__":
    args = parser.parse_args()
    args.device = 'cuda' if torch.cuda.is_available() else 'cpu'

    torch.manual_seed(args.seed)
    if args.device=='cuda':
        torch.cuda.manual_seed(args.seed)
    random.seed(args.seed)

    kwargs = {'num_workers': 2, 'pin_memory': True} if args.device=='cuda' else {}
    run(args, kwargs)

