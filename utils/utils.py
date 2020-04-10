import torch
import os

def importing_model(args):
    if args.model_name == 'vae':
        from models.VAE import VAE
    elif args.model_name == 'hvae_2level':
        from models.HVAE_2level import VAE
    elif args.model_name == 'convhvae_2level':
        from models.convHVAE_2level import VAE
    elif args.model_name == 'new_vae':
        from models.new_vae import VAE
    elif args.model_name == 'single_conv':
        from models.fully_conv import VAE
    else:
        raise Exception('Wrong name of the model!')
    return VAE


def save_model(save_path, load_path, content):
    torch.save(content, save_path)
    os.rename(save_path, load_path)


def load_model(load_path, model, optimizer=None):
    checkpoint = torch.load(load_path)
    model.load_state_dict(checkpoint['state_dict'])
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint['optimizer'])
    return checkpoint
