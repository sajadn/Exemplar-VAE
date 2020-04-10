import os
from torchvision import datasets
import numpy as np
from scipy.io import loadmat
from .base_load_data import base_load_data


class dynamic_mnist_loader(base_load_data):
    def __init__(self, args, use_fixed_validation=False, no_binarization=False):
        super(dynamic_mnist_loader, self).__init__(args, use_fixed_validation, no_binarization=no_binarization)

    def obtain_data(self):
        train = datasets.MNIST(os.path.join('datasets', self.args.dataset_name), train=True, download=True)
        test = datasets.MNIST(os.path.join('datasets', self.args.dataset_name), train=False)
        return train, test


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
        train_loader, val_loader, test_loader, args = cifar10_loader(args).load_dataset(**kwargs)
    else:
        raise Exception('Wrong name of the dataset!')
    print('train size', len(train_loader.dataset))
    if val_loader is not None:
        print('val size', len(val_loader.dataset))
    print('test size', len(test_loader.dataset))
    return train_loader, val_loader, test_loader, args
