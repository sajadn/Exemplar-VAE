import torch
import torch.nn as nn
import os
import time
from utils.plot_images import imshow
import matplotlib.pylab as plt
import torchvision
from pylab import rcParams

rcParams['figure.figsize'] = 15, 15


def compute_accuracy(classifier, model, loader, mean, args, dir=None, plot_mistakes=False):
    acc = 0
    mistakes_list = []
    for data, labels in loader:
        try:
            if model.args.use_logit is True and model.args.continuous is True:
                data = torch.round(model.logit_inverse(data) * 255) / 255
        except:
            pass
        labels = labels.to(args.device)
        pred = classifier(data.double().to(args.device) - mean)
        acc += torch.mean((labels == torch.argmax(pred, dim=1)).double())
        mistakes = (labels != torch.argmax(pred, dim=1))
        mistakes_list.append(data[mistakes])
    mistakes_list = torch.cat(mistakes_list, dim=0)
    if plot_mistakes is True:
        imshow(torchvision.utils.make_grid(mistakes_list.reshape(-1, *args.input_size)))
        # plt.show()
        plt.axis('off')
        plt.savefig(os.path.join(dir, 'mistakes.png'), bbox_inches='tight')
    acc /= len(loader)
    return acc


def save_model(save_path, load_path, content):
    torch.save(content, save_path)
    os.rename(save_path, load_path)


def load_model(load_path, model, optimizer=None):
    checkpoint = torch.load(load_path)
    model.load_state_dict(checkpoint['state_dict'])
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint['optimizer'])
    return checkpoint


def compute_loss(pred, label, args):
    held_out_percent = 0.1

    denom = torch.logsumexp(pred, dim=1, keepdim=True)
    prediction = pred - denom

    one_hot_label = torch.ones_like(prediction) * (held_out_percent / 10)
    one_hot_label[torch.arange(args.batch_size), label] += (1 - held_out_percent)
    return -torch.sum(prediction * one_hot_label, dim=1).mean()


def classify_data(train_loader, val_loader, test_loader, dir, args, model):
    classifier = nn.Sequential(nn.Linear(784, args.hidden_units), nn.ReLU(),
                               nn.Linear(args.hidden_units, args.hidden_units), nn.ReLU(),
                               nn.Linear(args.hidden_units, 10)).double().to(args.device)

    lr = args.lr

    optimizer = torch.optim.SGD(classifier.parameters(), lr=lr, momentum=0.9)
    epochs = args.epochs
    mean = 0
    lr_lambda = lambda epoch: 1-(0.99)*(epoch/epochs)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)
    os.makedirs(dir, exist_ok=True)

    if os.path.exists(os.path.join(dir, 'checkpoint.pth')):
        checkpoint = load_model(os.path.join(dir, 'checkpoint.pth'),
                                model=classifier,
                                optimizer=optimizer)
        begin_epoch = checkpoint['epoch']
    else:
        begin_epoch = 1

    for epoch_number in range(begin_epoch, epochs + 1):
        start_time = time.time()
        if epoch_number % 10 == 0:
            content = {'epoch': epoch_number, 'state_dict': classifier.state_dict(),
                       'optimizer': optimizer.state_dict()}
            save_model(os.path.join(dir, 'checkpoint_temp.pth'),
                                              os.path.join(dir, 'checkpoint.pth'), content)

        print('epoch number:', epoch_number)
        for index, data in enumerate(train_loader):

            data, _, label = data
            data_augment = model.reference_based_generation_x(reference_image=data.detach(), N=1).squeeze().double()
            label_augment = label

            data_augment = data_augment.to(args.device)
            label_augment = label_augment.to(args.device)

            data  = data.to(args.device).double()
            label = label.to(args.device).long()

            # imshow(torchvision.utils.make_grid(data.reshape(-1, *args.input_size)).detach())
            # plt.show()
            try:
                if model.args.use_logit is True and model.args.continuous is True:
                    data = torch.round(model.logit_inverse(data) * 255) / 255
            except:
                pass
            data_augment = torch.round(data_augment * 255) / 255

            loss1 = compute_loss(classifier(data), label, args)
            loss2 = compute_loss(classifier(data_augment), label_augment, args)

            loss = args.hyper_lambda*loss1 + (1-args.hyper_lambda)*loss2

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        scheduler.step(epoch=epoch_number)

        for param_group in optimizer.param_groups:
            print('learning rate:', param_group['lr'])
            break

        if val_loader is not None:
            val_acc = compute_accuracy(classifier, model, val_loader, mean, args)
            print('val acc', val_acc.item())
        test_acc = compute_accuracy(classifier, model, test_loader, mean, args)
        print('accuracy test:', test_acc.item())
        print("time:", time.time() - start_time)

    content = {'epoch': args.epochs, 'state_dict': classifier.state_dict(),
               'optimizer': optimizer.state_dict()}
    save_model(os.path.join(dir, 'checkpoint_temp.pth'), os.path.join(dir, 'checkpoint.pth'), content)
    classifier.eval()
    if val_loader is not None:
        val_acc = compute_accuracy(classifier, model, val_loader, mean, args)
        print('accuracy val:', val_acc.item())
    else:
        val_acc = torch.zeros(1)
    test_acc = compute_accuracy(classifier, model, test_loader, mean, args, dir=dir, plot_mistakes=True)
    print('accuracy test:', test_acc.item())
#
#
    return (test_acc*10000).item()/100, (val_acc*10000).item()/100
