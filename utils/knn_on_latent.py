import torch


def find_nearest_neighbors(z_val, z_train, z_train_log_var):
    z_expand = z_val.unsqueeze(1)
    means = z_train.unsqueeze(0)
    distance = (z_expand - means)**2
    _, indices_batch = (torch.sum(distance, dim=2)**(0.5)).topk(k=20, dim=1, largest=False, sorted=True)
    return indices_batch


def extract_full_data(data_loader):
    full_data = []
    full_labels = []
    full_indices = []
    for data in data_loader:
        if len(data) == 3:
            data, indices, labels = data
            full_indices.append(indices)
        else:
            data, labels = data
        full_data.append(data)
        full_labels.append(labels)
    full_data = torch.cat(full_data, dim=0)
    full_labels = torch.cat(full_labels, dim=0)
    if len(full_indices) > 0:
        full_indices = torch.cat(full_indices, dim=0)
    return full_data, full_indices, full_labels


# TODO refactor this fucntion
def report_knn_on_latent(train_loader, val_loader, test_loader, model, dir, knn_dictionary, args, val=True):
    train_data, _, train_labels = extract_full_data(train_loader)
    val_data, _, val_labels = extract_full_data(val_loader)
    test_data, _, test_labels = extract_full_data(test_loader)

    train_data = train_data.to(args.device)
    val_data = val_data.to(args.device)

    if val is True:
        data_to_evaluate = val_data
        labels = val_labels
    else:
        train_data = torch.cat((train_data, val_data), dim=0)
        train_labels = torch.cat((train_labels, val_labels), dim=0)
        data_to_evaluate = test_data
        labels = test_labels

    with torch.no_grad():
        z_train = []
        for i in range(len(train_data)//args.batch_size):
            train_batch = train_data[i*args.batch_size: (i+1)*args.batch_size]
            z_train_batch, _ = model.q_z(train_batch.to(args.device), prior=True)
            z_train.append(z_train_batch)
        z_train = torch.cat(z_train, dim=0)

    print(z_train.shape)
    indices = []
    for i in range(len(data_to_evaluate)//args.batch_size):
        z_val, _ = model.q_z(data_to_evaluate[i*args.batch_size: (i+1)*args.batch_size].to(args.device), prior=True)
        indices.append(find_nearest_neighbors(z_val, z_train, None))
    indices = torch.cat(indices, dim=0)

    for k in knn_dictionary.keys():
        k = int(k)
        k_labels = train_labels[indices[:, :k]].squeeze().long()
        num_classes = 10
        counts = torch.zeros(len(test_loader.dataset), num_classes)
        for i in range(num_classes):
            counts[:, i] = (k_labels == torch.tensor(i).long()).sum(dim=1)
        y_pred = torch.argmax(counts, dim=1)
        acc = (torch.mean((y_pred == labels.long()).float()) * 10000).round().item()/100
        print('K:', k, 'Accuracy:', acc)
        knn_dictionary[str(k)].append(acc)
