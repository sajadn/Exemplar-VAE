import os
from torchvision import datasets, transforms
import numpy as np
from scipy.io import loadmat
from .base_load_data import base_load_data
import wget
import PIL
import h5py
import torch
import torch.utils.data as data_utils
import zipfile
from utils.utils import scaled_logit
from PIL import Image

class dynamic_mnist_loader(base_load_data):
    def __init__(self, args, use_fixed_validation=False, no_binarization=False):
        super(dynamic_mnist_loader, self).__init__(args, use_fixed_validation, no_binarization=no_binarization)

    def obtain_data(self):
        train = datasets.MNIST(os.path.join('datasets', self.args.dataset_name), train=True, download=True)
        test = datasets.MNIST(os.path.join('datasets', self.args.dataset_name), train=False)
        return train, test


class h5dataset(torch.utils.data.Dataset):
    def __init__(self, in_file, args):
        super(h5dataset, self).__init__()
        self.args = args

        self.file = h5py.File(in_file, 'r')
        self.n_images, self.nx, self.ny, self.ch = self.file['data'].shape
        self.tensors = (self.file['data'], torch.arange(self.n_images).reshape(-1, 1))
        self.trns = transforms.Compose([
                                      transforms.Pad(4, padding_mode='reflect'),
                                      transforms.RandomCrop(64),
                                      transforms.RandomHorizontalFlip(0.5)])

    def __getitem__(self, index):
        return self.preprocess(self.tensors[0][index, :, :, :].astype('float32')),\
               self.tensors[1][index], torch.tensor([])


    def __len__(self):
        return self.n_images

    def preprocess(self, data):
        if self.args.input_type == 'gray' or self.args.input_type == 'continuous':
            # if self.args.with_augmentation and len(self.tensors[0]) > 50000 and data.shape[0] == 3:
            #     data = np.transpose(data, (1, 2, 0))
            #     data = self.trns(Image.fromarray(np.uint8(data)))
            #     data = np.asarray(data)
            #     data = np.transpose(data, (2, 0, 1))

            data = np.clip((data + 0.5) / 256., 0., 1.)
            if self.args.use_logit:
                data = scaled_logit(data, self.args.lambd)
            elif self.args.zero_center:
                data -= 0.5

        else:
            data = data / 255.
        return  torch.from_numpy(np.reshape(data, (-1, np.prod(self.args.input_size)))).float()


class celebA_loader(base_load_data):
    def __init__(self, args, use_fixed_validation=False, no_binarization=False):
        super(celebA_loader, self).__init__(args, use_fixed_validation, no_binarization=no_binarization)

    def obtain_data(self):
        zip_file = 'CelebA.zip'
        if not os.path.exists(os.path.join(os.path.join('datasets', self.args.dataset_name), 'train.h5')):
            url = "http://www.cs.toronto.edu/~sajadn/CelebA.zip"
            wget.download(url, zip_file)
            with zipfile.ZipFile(zip_file, 'r') as zip_ref:
                zip_ref.extractall(os.path.join(os.path.join('datasets', self.args.dataset_name)))

        train = h5dataset(os.path.join(os.path.join('datasets', self.args.dataset_name), 'train.h5'), self.args)
        valid = h5dataset(os.path.join(os.path.join('datasets', self.args.dataset_name), 'valid.h5'), self.args)
        test = h5dataset(os.path.join(os.path.join('datasets', self.args.dataset_name), 'test.h5'), self.args)
        # whole_temp = []
        # for i in range(len(test)):
        #     print(i)
        #     temp = test[i]
        #     temp = np.transpose(temp, (1, 2, 0))
        #     whole_temp.append(temp)
        # np.stack(whole_temp, axis=0)
        # h5f = h5py.File('data.h5', 'w')
        # h5f.create_dataset('data', data=whole_temp)
        # h5f.close()
        # exit()
        # train = datasets.CelebA(os.path.join('datasets', self.args.dataset_name), split='valid', download=True)
        # self.center_crop_resize(train)
        # test = datasets.MNIST(os.path.join('datasets', self.args.dataset_name), train=False)


        return (train, valid), test


    def post_processing(self, x_train, x_val, x_test, y_train, y_val, y_test, init_mean=0.05, init_std=0.01, **kwargs):
        train_loader = data_utils.DataLoader(x_train, batch_size=self.args.batch_size, shuffle=True, **kwargs)
        if len(x_val) > 0:
            val_loader = data_utils.DataLoader(x_val, batch_size=self.args.test_batch_size, shuffle=True, **kwargs)
        else:
            val_loader = None
        test_loader = data_utils.DataLoader(x_test, batch_size=self.args.test_batch_size, shuffle=False, **kwargs)
        self.vampprior_initialization(x_train, init_mean, init_std)
        return train_loader, val_loader, test_loader


    def center_crop_resize(self, dataset):
        whole_dataset = []
        for index in range(len(dataset.filename)):
            print(index)
            transform = transforms.Compose([
                # transforms.ToTensor(),
                transforms.CenterCrop(140),
                transforms.Resize(64)])

            X = PIL.Image.open(os.path.join(dataset.root, dataset.base_folder, "img_align_celeba", dataset.filename[index]))
            if transform is not None:
                X = transform(X)
            whole_dataset.append(np.transpose(np.asanyarray(X), (2, 0, 1)))
            # X.save(os.path.join(dataset.root, dataset.base_folder, "img_align_celeba", dataset.filename[index]), "JPEG")

        # return (train, valid), test
        whole_dataset = np.stack(whole_dataset, axis=0)
        print(whole_dataset.shape)
        h5f = h5py.File('data.h5', 'w')
        h5f.create_dataset('data', data=whole_dataset)
        h5f.close()

    def preprocessing_(self, data):
        return data

    def seperate_data_from_label(self, train_dataset, test_dataset):
        return train_dataset, (None, None), test_dataset, None

class fashion_mnist_loader(base_load_data):
    def __init__(self, args, use_fixed_validation=False, no_binarization=False):
        super(fashion_mnist_loader, self).__init__(args, use_fixed_validation, no_binarization=no_binarization)

    def obtain_data(self):
        train = datasets.FashionMNIST(os.path.join('datasets', self.args.dataset_name), train=True, download=True)
        test = datasets.FashionMNIST(os.path.join('datasets', self.args.dataset_name), train=False)
        return train, test


class svhn_loader(base_load_data):
    def __init__(self, args, use_fixed_validation=False, no_binarization=False):
        super(svhn_loader, self).__init__(args, use_fixed_validation, no_binarization=no_binarization)

    def obtain_data(self):
        train = datasets.SVHN(os.path.join('datasets', self.args.dataset_name), split='train', download=True)
        test = datasets.SVHN(os.path.join('datasets', self.args.dataset_name), split='test', download=True)
        return train, test

    def seperate_data_from_label(self, train_dataset, test_dataset):
        x_train = train_dataset.data
        y_train = train_dataset.labels.astype(dtype=int)
        x_test = test_dataset.data
        y_test = test_dataset.labels.astype(dtype=int)
        return x_train, y_train, x_test, y_test


class static_mnist_loader(base_load_data):
    def __init__(self, args, use_fixed_validation=False, no_binarization=False):
        super(static_mnist_loader, self).__init__(args, use_fixed_validation, no_binarization=no_binarization)

    def obtain_data(self):
        def lines_to_np_array(lines):
            return np.array([[int(i) for i in line.split()] for line in lines])

        with open(os.path.join('datasets', self.args.dataset_name, 'binarized_mnist_train.amat')) as f:
            lines = f.readlines()
        x_train = lines_to_np_array(lines).astype('float32')
        with open(os.path.join('datasets', self.args.dataset_name, 'binarized_mnist_valid.amat')) as f:
            lines = f.readlines()
        x_val = lines_to_np_array(lines).astype('float32')
        with open(os.path.join('datasets', self.args.dataset_name, 'binarized_mnist_test.amat')) as f:
            lines = f.readlines()
        x_test = lines_to_np_array(lines).astype('float32')

        y_train = np.zeros((x_train.shape[0], 1)).astype(int)
        y_val = np.zeros((x_val.shape[0], 1)).astype(int)
        y_test = np.zeros((x_test.shape[0], 1)).astype(int)
        return (x_train, x_val, y_train, y_val), (x_test, y_test)

    def seperate_data_from_label(self, train_dataset, test_dataset):
        x_train, x_val, y_train, y_val = train_dataset
        x_test, y_test = test_dataset
        return (x_train, x_val), (y_train, y_val), x_test, y_test

    def preprocessing_(self, x_train, x_test):
        return x_train, x_test


class omniglot_loader(base_load_data):
    def __init__(self, args, use_fixed_validation=False, no_binarization=False):
        super(omniglot_loader, self).__init__(args, use_fixed_validation, no_binarization=no_binarization)

    def obtain_data(self):
        def reshape_data(data):
            return data.reshape((-1, 28, 28)).reshape((-1, 28*28), order='F')
        dataset_file = os.path.join('datasets', self.args.dataset_name, 'chardata.mat')
        if not os.path.exists(dataset_file):
            url = "https://raw.githubusercontent.com/yburda/iwae/master/datasets/OMNIGLOT/chardata.mat"
            wget.download(url, dataset_file)

        omni_raw = loadmat(os.path.join('datasets', self.args.dataset_name, 'chardata.mat'))

        x_train = reshape_data(omni_raw['data'].T.astype('float32'))
        x_test = reshape_data(omni_raw['testdata'].T.astype('float32'))

        y_train = omni_raw['targetchar'].reshape((-1, 1))
        y_test = omni_raw['testtargetchar'].reshape((-1, 1))
        return (x_train, y_train), (x_test, y_test)

    def seperate_data_from_label(self, train_dataset, test_dataset):
        x_train, y_train = train_dataset
        x_test, y_test = test_dataset
        return x_train, y_train, x_test, y_test

    def preprocessing_(self, x_train, x_test):
        return x_train, x_test


class cifar10_loader(base_load_data):
    def __init__(self, args, use_fixed_validation=False, no_binarization=False):
        super(cifar10_loader, self).__init__(args, use_fixed_validation, no_binarization=no_binarization)

    def obtain_data(self):
        training_dataset = datasets.CIFAR10(os.path.join('datasets', self.args.dataset_name), train=True, download=True)
        test_dataset = datasets.CIFAR10(os.path.join('datasets', self.args.dataset_name), train=False)
        return training_dataset, test_dataset

    def seperate_data_from_label(self, train_dataset, test_dataset):
        train_data = np.swapaxes(np.swapaxes(train_dataset.data, 1, 2), 1, 3)
        y_train = np.zeros((train_data.shape[0], 1)).astype(int)
        test_data = np.swapaxes(np.swapaxes(test_dataset.data, 1, 2), 1, 3)
        y_test = np.zeros((test_data.shape[0], 1)).astype(int)
        return train_data, y_train, test_data, y_test


def load_dataset(args, training_num=None, use_fixed_validation=False, no_binarization=False, **kwargs):
    if training_num is not None:
        args.training_set_size = training_num
    if args.dataset_name == 'static_mnist':
        args.input_size = [1, 28, 28]
        args.input_type = 'binary'
        train_loader, val_loader, test_loader, args = static_mnist_loader(args).load_dataset(**kwargs)
    elif args.dataset_name == 'dynamic_mnist':
        if training_num is None:
            args.training_set_size = 50000
        args.input_size = [1, 28, 28]
        if args.continuous is True:
            args.input_type = 'gray'
            args.dynamic_binarization = False
            no_binarization = True
        else:
            args.input_type = 'binary'
            args.dynamic_binarization = True

        train_loader, val_loader, test_loader, args = \
            dynamic_mnist_loader(args, use_fixed_validation, no_binarization=no_binarization).load_dataset(**kwargs)
    elif args.dataset_name == 'fashion_mnist':
        if training_num is None:
            args.training_set_size = 50000
        args.input_size = [1, 28, 28]

        if args.continuous is True:
            print("*****Continuous Data*****")
            args.input_type = 'gray'
            args.dynamic_binarization = False
            no_binarization = True
        else:
            args.input_type = 'binary'
            args.dynamic_binarization = True

        train_loader, val_loader, test_loader, args = \
            fashion_mnist_loader(args, use_fixed_validation, no_binarization=no_binarization).load_dataset(**kwargs)
    elif args.dataset_name == 'omniglot':
        if training_num is None:
            args.training_set_size = 23000
        args.input_size = [1, 28, 28]
        args.input_type = 'binary'
        args.dynamic_binarization = True
        train_loader, val_loader, test_loader, args = omniglot_loader(args).load_dataset(**kwargs)
    elif args.dataset_name == 'svhn':
        args.training_set_size = 60000
        args.input_size = [3, 32, 32]
        args.input_type = 'continuous'
        train_loader, val_loader, test_loader, args = svhn_loader(args).load_dataset(**kwargs)
    elif args.dataset_name == 'cifar10':
        args.training_set_size = 40000
        args.input_size = [3, 32, 32]
        args.input_type = 'continuous'
        args.lambd = 0.05
        train_loader, val_loader, test_loader, args = cifar10_loader(args).load_dataset(**kwargs)
    elif args.dataset_name == 'CelebA':
        args.training_set_size = 162770
        args.input_size = [3, 64, 64]
        args.input_type = 'continuous'
        args.lambd = 0.0001
        train_loader, val_loader, test_loader, args = celebA_loader(args).load_dataset(**kwargs)
    else:
        raise Exception('Wrong name of the dataset!')
    print('train size', len(train_loader.dataset))
    if val_loader is not None:
        print('val size', len(val_loader.dataset))
    print('test size', len(test_loader.dataset))
    return train_loader, val_loader, test_loader, args
