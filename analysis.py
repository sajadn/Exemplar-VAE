import torch
import argparse
from utils.load_data.data_loader_instances import load_dataset
import torchvision
import matplotlib.pyplot as plt
import numpy as np
import os
from utils.plot_images import imshow
from utils.utils import load_model
from utils.classify_data import classify_data
from utils.knn_on_latent import report_knn_on_latent, extract_full_data
from utils.evaluation import compute_mean_variance_per_dimension
from utils.plot_images import plot_images_in_line, generate_fancy_grid
from utils.utils import importing_model
from sklearn.manifold import TSNE
import copy
from pylab import rcParams
from utils.utils import inverse_scaled_logit, scaled_logit_torch

parser = argparse.ArgumentParser(description='VAE+VampPrior')
parser.add_argument('--KNN', action='store_true', default=False, help='run KNN classification on latent')
parser.add_argument('--generate', action='store_true', default=False, help='generate images')
parser.add_argument('--classify', action='store_true', default=False,
                    help='train a classifier on data with augmentation')
parser.add_argument('--dir', type=str, default='directory of pretrained model')
parser.add_argument('--just_log_likelihood', action='store_true', default=False)
parser.add_argument('--cyclic_generation', action='store_true', default=False, help='cyclic generation')
parser.add_argument('--training_set_size', default=50000, type=int)
parser.add_argument('--hyper_lambda', type=float, default=0.4, help='proportion of real data to augmented data')
parser.add_argument('--lr', type=float, default=0.1)
parser.add_argument('--batch_size', type=int, default=100)
parser.add_argument('--input_size', type=list, default=[1, 28, 28])
parser.add_argument('--count_active_dimensions', action='store_true', default=False)
parser.add_argument('--grid_interpolation', action='store_true', default=False)
parser.add_argument('--tsne_visualization', action='store_true', default=False)
parser.add_argument('--hidden_units', type=int, default=1024)
parser.add_argument('--save_model_path', type=str, default='')
parser.add_argument('--classification_dir', type=str, default='classification_report')
parser.add_argument('--epochs', type=int, default=100)
parser.add_argument('--seed', type=int, default=1)
parser.add_argument('--interpolate', action='store_true', default=False)
parser.add_argument('--generate_fid', action='store_true', default=False)
parser.add_argument('--scale_std', type=float, default=1.)

args = parser.parse_args()

print(args)

TRAIN_NUM = 50000


def plot_data(data, labels):
    k = 10
    print(data.shape)
    subplot_num = data.shape[1]
    for i in range(subplot_num):
        plt.subplot2grid((subplot_num, 1), (i, 0), colspan=1, rowspan=1)
        imshow(torchvision.utils.make_grid(data[:k, i, :].view(-1, 1, 28, 28)))
        plt.axis('off')
        print(labels[:k, i, :].squeeze())
    plt.show()

directory = args.dir


def grid_interpolation_in_latent(model, dir, index, reference_image):
    z, _ = model.q_z(reference_image.to(args.device), prior=True)
    whole_generation = []
    for offset_0 in range(-2, 3, 1):
        row_generation = []
        for offset_1 in range(-2, 3, 1):
            new_z = copy.deepcopy(z)
            new_z[0][0] += offset_0*3
            new_z[0][1] += offset_1*3
            image = model.generate_x_from_z(new_z, with_reparameterize=False)
            row_generation.append(image)
        whole_generation.append(torch.cat(row_generation, dim=0))
        # print("LENNN", len(whole_generation))
    whole_generation = torch.cat(whole_generation, dim=0)
    print('whole_generation shape', whole_generation.shape)
    imshow(torchvision.utils.make_grid(whole_generation.reshape(-1, *model.args.input_size), nrow=5))
    save_dir = os.path.join(dir, 'grid_interpolation')
    os.makedirs(save_dir, exist_ok=True)
    plt.axis('off')
    plt.savefig(os.path.join(save_dir, 'interpolation{}'.format(i)), bbox='tight')


def interpolation(model, reference_images, dir, steps=20, k=0):
    z, _ = model.q_z(reference_images)
    first = z[0]
    second = z[1]
    direction = (second-first)/steps
    generated_list = []
    for i in range(steps):
        generated = model.generate_x_from_z(first.unsqueeze(0), with_reparameterize=False)
        generated_list.append(generated)
        first += direction

    save_dir = os.path.join(dir, 'interpolation')
    os.makedirs(save_dir, exist_ok=True)
    plot_images_in_line(torch.cat(generated_list, dim=0), config, dir=save_dir, file_name='interpolation_{}.png'.format(k))


def compute_test_metrics(test_log_likelihood, test_kl, test_re):
    test_log_likelihood.append(torch.load(dir + model_name + '.test_log_likelihood'))

    kl = torch.load(dir + model_name + '.test_kl')
    if type(kl) == torch.Tensor:
        kl = kl.cpu().numpy()
    test_kl.append(kl)

    reconst = torch.load(dir + model_name + '.test_re')
    if type(reconst) == torch.Tensor:
        reconst = reconst.cpu().numpy()
    test_re.append(reconst)


def cyclic_generation(start_data, dir, index):
    cyclic_generation_dir = os.path.join(dir, 'cyclic_generation')
    os.makedirs(cyclic_generation_dir, exist_ok=True)
    single_data = start_data.unsqueeze(0)
    generated_cycle = [single_data.to(args.device)]
    for i in range(29):
        single_data = \
            model.reference_based_generation_x(N=1, reference_image=single_data)
        generated_cycle.append(single_data)

    generated_cycle = torch.cat(generated_cycle, dim=0)
    plot_images_in_line(generated_cycle, args, cyclic_generation_dir, 'cycle_{}.png'.format(index))


temp = ''
active_units_text = ''
args.device = 'cuda' if torch.cuda.is_available() else 'cpu'

for folder in sorted(os.listdir(directory)):
    if os.path.isdir(directory+'/'+folder) is False:
        continue
    knn_results = []
    test_log_likelihoods, test_kl, test_reconst, active_dimensions = [], [], [], []
    knn_dictionary = {'3': [], '5': [], '7': [], '9': [], '11': [], '13': [], '15': []}


    torch.manual_seed(args.seed)
    if args.device=='cuda':
        torch.cuda.manual_seed(args.seed)
    np.random.seed(args.seed)

    for filename in os.listdir(directory+'/'+folder):
        print('filename**', filename)
        dir = directory + '/' + folder+'/'+filename + '/'
        model_name_start_index = folder.find('model_name=')
        model_name = folder[model_name_start_index + len('model_name='):]
        print("MODEL NAME", model_name)

        config = torch.load(dir + model_name + '.config')
        config.device = args.device
        config.scale_std = args.scale_std
        VAE = importing_model(config)
        model = VAE(config)
        model.to(args.device)
        print(config)
        train_loader, val_loader, test_loader, config = load_dataset(config,
                                                                     training_num=args.training_set_size,
                                                                     no_binarization=True)

        if args.just_log_likelihood is False:
            load_model(dir + 'checkpoint_best.pth', model)
            model.eval()
            try:
                print('prior variance', model.prior_log_variance.item())
            except:
                pass

            if args.cyclic_generation:
                with torch.no_grad():
                    for i in range(10):
                        random_image = torch.rand([784])
                        cyclic_generation(random_image, dir, index=i)

            if args.KNN:
                with torch.no_grad():
                    report_knn_on_latent(train_loader, val_loader, test_loader, model,
                                         dir, knn_dictionary, args, val=False)
            if args.generate:
                with torch.no_grad():
                    exemplars_n = 10
                    # selected_indices = torch.sort(torch.randint(low=0, high=config.training_set_size, size=(exemplars_n,)))[0]
                    selected_indices = torch.tensor([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])+350
                    reference_images, indices, labels =train_loader.dataset[selected_indices.numpy()]
                    per_exemplar = 11
                    generated = model.reference_based_generation_x(N=per_exemplar, reference_image=reference_images)
                    generated = generated.reshape(-1, per_exemplar, *config.input_size)
                    rcParams['figure.figsize'] = 4, 3
                    generated_dir = dir + 'generated/'
                    if config.use_logit:
                        reference_images = torch.floor(inverse_scaled_logit(reference_images, config.lambd)*256).int()
                    else:
                        reference_images = (reference_images*256).int()
                        generated = (generated*256).int()

                    generate_fancy_grid(config, dir, reference_images, generated, col_num=3, row_num=3)
            if args.generate_fid:
                index = 0
                with torch.no_grad():
                    for data, _, _ in train_loader:
                        if index>10000:
                            break
                        if config.prior == 'standard':
                            generated = model.generate_x(N=100).reshape(100, 3, 64, 64)
                        else:
                            generated = model.reference_based_generation_x(N=1, reference_image=data)

                        generated = generated.reshape(-1, *config.input_size)
                        generated_dir = dir + 'fid/'

                        for g in generated:
                            index += 1
                            # if read:
                            # g = g.reshape(3, 64, 64)
                            plt.imsave(arr=np.transpose(g.cpu().numpy(), (1, 2, 0)),
                                       fname=generated_dir + "generated_{}.jpg".format(index),
                                       cmap='gray', format='jpg')


            if args.count_active_dimensions:
                train_loader, val_loader, test_loader, config = load_dataset(config,
                                                                             training_num=args.training_set_size,
                                                                             no_binarization=False)
                with torch.no_grad():
                    num_active = compute_mean_variance_per_dimension(args, model, test_loader)
                    active_dimensions.append(num_active)

            #TODO remove loop
            if args.grid_interpolation:
                with torch.no_grad():
                    for i in range(100):
                        image = train_loader.dataset.tensors[0][torch.randint(low=0, high=args.training_set_size,
                                                                              size=(1,))]
                        grid_interpolation_in_latent(model, dir, i, reference_image=image)

            if args.interpolate:
                for k in range(100):
                    indices = torch.randint(low=0, high=args.training_set_size, size=(2,))
                    reference_images = train_loader.dataset[torch.sort(indices)[0].numpy()][0]
                    interpolation(model, reference_images.to(args.device), dir, steps=8, k=k)


            if args.tsne_visualization:
                rcParams['figure.figsize'] = 10, 10
                test_x, _, test_labels = extract_full_data(test_loader)
                test_z, _ = model.q_z(test_x.to(args.device))
                tsne = TSNE(n_components=2)
                plt_colors = np.array(
                    ['blue', 'orange', 'green', 'red', 'cyan', 'pink', 'purple', 'brown', 'gray', 'olive'])

                if config.dataset_name == 'fashion_mnist':
                    label_names = ['T-shirt', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt',
                                   'Sneaker', 'Bag', 'Ankle Boot']
                elif config.dataset_name == 'dynamic_mnist':
                    label_names = ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10']
                else:
                    label_names = None

                points_to_visualize = tsne.fit_transform(X=test_z.detach().cpu().numpy())
                for i in range(len(label_names)):
                    label_i = (test_labels == i)
                    plt.scatter(points_to_visualize[label_i, 0], points_to_visualize[label_i, 1],
                                c=plt_colors[i], s=3, label=label_names[i])
                plt.legend(fontsize=10, markerscale=5)
                plt.savefig(dir+'tsne.png')
                plt.show()

            if args.classify:
                test_acc = []
                val_acc = []
                test_acc_single_run, val_acc_single_run = classify_data(train_loader, val_loader, test_loader,
                                                                        args.classification_dir, args, model)
                test_acc.append(test_acc_single_run)
                val_acc.append(val_acc_single_run)
                test_acc = np.array(test_acc)
                val_acc = np.array(val_acc)

                print('averaged test accuracy: {0:.2f} \\pm {1:.2f}'.format(np.mean(test_acc), np.std(test_acc)))
                print('averaged val accuracy: {0:.2f} \\pm {1:.2f}'.format(np.mean(val_acc), np.std(val_acc)))
                exit()
        else:
            compute_test_metrics(test_log_likelihoods, test_kl, test_reconst)

    if args.just_log_likelihood:
        test_log_likelihoods = np.array(test_log_likelihoods)
        print("test log-likelihood", np.mean(test_log_likelihoods), np.std(test_log_likelihoods))

    if args.count_active_dimensions:
        active_dimensions = np.array(active_dimensions).astype(float)
        print(np.mean(active_dimensions), np.std(active_dimensions))
