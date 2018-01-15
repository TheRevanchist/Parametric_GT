import os
import pickle
import gtg
import utils
import utils2
import FileDataset
import torch.nn as nn
import torch.optim as optim
from net import train, evaluate
import torch
from pathlib2 import Path
import torchvision


def main2():
    root = '.'
    current_dataset = 'caltech'
    out_dir = os.path.join(root, 'out', current_dataset)
    feature_dir = os.path.join(out_dir, 'feature_data')
    net_dir = os.path.join(out_dir, 'net')
    gtg_labels_dir = os.path.join(out_dir, 'gtg_labels')
    nr_classes = 256

    for pkl_name in os.listdir(feature_dir):
        with open(os.path.join(feature_dir, pkl_name), 'rb') as pkl:
            net_name, labels, features, fnames = pickle.load(pkl)

        W = Path('W.pickle')
        if W.exists():
            with open('W.pickle', 'rb') as f:
                W = pickle.load(f)
        else:
            W = gtg.sim_mat(features)
            with open('W.pickle', 'wb') as f:
                pickle.dump(W, f, pickle.HIGHEST_PROTOCOL)
        # W = gtg.sim_mat(features)
        nr_objects = features.shape[0]
        labelled, unlabelled = utils2.create_mapping(nr_objects, 0.1)
        ps = utils2.gen_init_rand_probability(labels, labelled, unlabelled, nr_classes)
        gtg_accuracy, Ps_new = utils2.get_accuracy(W, ps, labels, labelled, unlabelled, len(unlabelled))
        gtg_labels = Ps_new.argmax(axis=1)

        nname, ind = pkl_name.split('_')

        names_folds = os.listdir('Datasets/caltech/train_0')
        names_folds.sort()
        gtg_label_file = os.path.join(gtg_labels_dir, nname + '.txt')
        utils2.gen_gtg_label_file(fnames, names_folds, gtg_labels, gtg_label_file)

        stats = (.517, .5015, .4736, .315, .3111, .324)
        transform = torchvision.transforms.Compose([
            torchvision.transforms.RandomSizedCrop(224),
            torchvision.transforms.RandomHorizontalFlip(),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(mean=(stats[0], stats[1], stats[2]),
                                             std=(stats[3], stats[4], stats[5]))
        ])

        dataset = 'Datasets/' + current_dataset
        dataset_train = os.path.join(dataset, 'train_' + ind[0])
        dataset_test = os.path.join(dataset, 'test_' + ind[0])
        dataset = FileDataset.CaltechFileDataset(gtg_label_file, dataset_train, transform)

        max_epochs = 10
        batch_size = 8

        nets_and_features = utils.create_dict_nets_and_features()

        train_loader = utils.prepare_train_loader(dataset_train, stats, batch_size)
        test_loader = utils.prepare_val_loader(dataset_test, stats, batch_size)

        net, feature_size = utils.create_net(nr_classes, nets_and_features, net_type=nname)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(net.parameters(), lr=1e-4)

        best_net = train(net, nname, train_loader, test_loader, optimizer, criterion, max_epochs, net_dir, ind[0])

        net.load_state_dict(torch.load(best_net))
        net_accuracy = evaluate(net, test_loader)


if __name__ == '__main__':
    main2()