from __future__ import print_function
import torch
import torch.utils.data as data_utils
import numpy as np
from abc import ABC, abstractmethod


class base_load_data(ABC):
    def __init__(self, args, use_fixed_validation=False, no_binarization=False):
        self.args = args
        self.train_num = args.training_set_size
        self.use_fixed_validation = use_fixed_validation
        self.no_binarization = no_binarization

    @abstractmethod
    def obtain_data(self):
        pass

    def logit(self, x):
        return np.log(x) - np.log1p(-x)

    def seperate_data_from_label(self, train_dataset, test_dataset):
        x_train = train_dataset.data.numpy()
        y_train = train_dataset.train_labels.numpy().astype(int)
        x_test = test_dataset.data.numpy()
        y_test = test_dataset.test_labels.numpy().astype(int)
        return x_train, y_train, x_test, y_test

    def preprocessing_(self, x_train, x_test):
        if self.args.input_type == 'gray' or self.args.input_type == 'continuous':
            if self.args.use_logit:
                lambd = self.args.lambd
                x_train = self.logit(lambd + (1 - 2 * lambd) * (x_train + np.random.rand(*x_train.shape)) / 256.)
                x_test = self.logit(lambd + (1 - 2 * lambd) * (x_test + np.random.rand(*x_test.shape)) / 256.)
            elif self.args.continuous:
                x_train = np.clip((x_train + 0.5) / 256., 0., 1.)
                x_test = np.clip((x_test + 0.5) / 256., 0., 1.)
        else:
            x_train = x_train / 255.
            x_test = x_test / 255.

        return x_train, x_test

    def vampprior_initialization(self, x_train, init_mean, init_std):
        if self.args.use_training_data_init == 1:
            self.args.pseudoinputs_std = 0.01
            init = x_train[0:self.args.number_components].T
            self.args.pseudoinputs_mean = torch.from_numpy(
                init + self.args.pseudoinputs_std * np.random.randn(np.prod(self.args.input_size),
                                                               self.args.number_components)).float()
        else:
            self.args.pseudoinputs_mean = init_mean
            self.args.pseudoinputs_std = init_std

    def post_processing(self, x_train, x_val, x_test, y_train, y_val, y_test, init_mean=0.05, init_std=0.01, **kwargs):
        indices = np.arange(len(x_train)).reshape(-1, 1)
        train = data_utils.TensorDataset(torch.from_numpy(x_train).float(), torch.from_numpy(indices),
                                         torch.from_numpy(y_train))
        train_loader = data_utils.DataLoader(train, batch_size=self.args.batch_size, shuffle=True, **kwargs)

        if len(x_val) > 0:
            validation = data_utils.TensorDataset(torch.from_numpy(x_val).float(), torch.from_numpy(y_val))
            val_loader = data_utils.DataLoader(validation, batch_size=self.args.test_batch_size, shuffle=True, **kwargs)
        else:
            val_loader = None
        test = data_utils.TensorDataset(torch.from_numpy(x_test).float(), torch.from_numpy(y_test))
        test_loader = data_utils.DataLoader(test, batch_size=self.args.test_batch_size, shuffle=False, **kwargs)

        self.vampprior_initialization(x_train, init_mean, init_std)
        return train_loader, val_loader, test_loader

    def binarize(self, x_val, x_test):
        self.args.input_type = 'binary'
        np.random.seed(777)
        x_val = np.random.binomial(1, x_val)
        x_test = np.random.binomial(1, x_test)
        return x_val, x_test

    def load_dataset(self, **kwargs):
        # start processing
        train, test = self.obtain_data()
        x_train, y_train, x_test, y_test = self.seperate_data_from_label(train, test)
        x_train, x_test = self.preprocessing_(x_train, x_test)

        if self.use_fixed_validation is False:
            permutation = np.arange(len(x_train))
            np.random.shuffle(permutation)
            x_train = x_train[permutation]
            y_train = y_train[permutation]

        if self.args.dataset_name == 'static_mnist':
            x_train, x_val = x_train
            y_train, y_val = y_train
        else:
            x_val = x_train[self.train_num:]
            y_val = y_train[self.train_num:]
            x_train = x_train[:self.train_num]
            y_train = y_train[:self.train_num]

        # imshow(torchvision.utils.make_grid(torch.from_numpy(x_val[:50].reshape(-1, *self.args.input_size))))
        # plt.axis('off')
        # plt.show()

        x_train = np.reshape(x_train, (-1, np.prod(self.args.input_size)))
        x_val = np.reshape(x_val, (-1, np.prod(self.args.input_size)))

        x_test = np.reshape(x_test, (-1, np.prod(self.args.input_size)))

        if self.args.dynamic_binarization and self.no_binarization is False:
            x_val, x_test = self.binarize(x_val, x_test)

        print("data stats:")
        print(len(x_train), len(y_train))
        print(len(x_val), len(y_val))
        print(len(x_test), len(y_test))

        train_loader, val_loader, test_loader, = self.post_processing(x_train, x_val, x_test,
                                                                 y_train, y_val, y_test, **kwargs)

        return train_loader, val_loader, test_loader, self.args
