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
from split_dataset import gen_gtg_dataset, gen_labelled_dataset
from utils import prepare_loader_train, prepare_loader_val, create_net, create_dict_nets_and_features


def main2():
    root = '.'
    current_dataset = 'caltech'
    out_dir = os.path.join(root, 'out', current_dataset)
    feature_dir = os.path.join(out_dir, 'feature_data')
    feature_test_dir = os.path.join(out_dir, 'feature_data_test')
    net_dir = os.path.join(out_dir, 'net')
    gtg_labels_dir = os.path.join(out_dir, 'gtg_labels')
    only_labelled = os.path.join(out_dir, 'only_labelled')
    nr_classes = 256
    nets_and_features = create_dict_nets_and_features()
    results = 'results.txt'

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

        # now check gt in the testing set

        nname, ind = pkl_name.split('_')

        names_folds = os.listdir('Datasets/caltech/train_' + str(ind[0]))
        names_folds.sort()
        gtg_label_file = os.path.join(gtg_labels_dir, nname + '.txt')
        utils2.gen_gtg_label_file(fnames, names_folds, gtg_labels, gtg_label_file)

        # generate the new dataset
        gen_gtg_dataset('caltech/train_' + str(ind[0]), gtg_label_file, ind[0])

        stats = (.517, .5015, .4736, .315, .3111, .324)

        dataset = 'Datasets/' + current_dataset
        dataset_train = os.path.join(dataset, 'train_labelled_' + ind[0])
        dataset_test = os.path.join(dataset, 'test_' + ind[0])

        max_epochs = 1
        batch_size = 8

        train_loader = prepare_loader_train(dataset_train, stats, batch_size)
        test_loader = prepare_loader_val(dataset_test, stats, batch_size)

        net, feature_size = create_net(nr_classes, nets_and_features, net_type=nname)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(net.parameters(), lr=1e-4)

        trained_net = train(net, nname, train_loader, test_loader, optimizer, criterion, max_epochs, out_dir, ind[0])

        net.load_state_dict(torch.load(trained_net))
        net_accuracy = evaluate(net, test_loader) # toDO: store this
        print('Accuracy: ' + str(net_accuracy))

        # now check the accuracy of the net trained only in the labelled set
        label_file = os.path.join(only_labelled, nname + '.txt')
        utils2.gen_gtg_label_file(fnames, names_folds, gtg_labels, gtg_label_file)
        gen_labelled_dataset('caltech/train_' + str(ind[0]), label_file, ind[0])

        dataset_train = os.path.join(dataset, 'train_only_labelled_' + ind[0])

        train_loader = prepare_loader_train(dataset_train, stats, batch_size)
        test_loader = prepare_loader_val(dataset_test, stats, batch_size)

        net, feature_size = create_net(nr_classes, nets_and_features, net_type=nname)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(net.parameters(), lr=1e-4)

        trained_net = train(net, nname, train_loader, test_loader, optimizer, criterion, max_epochs, out_dir, ind[0])

        net.load_state_dict(torch.load(trained_net))
        net_accuracy = evaluate(net, test_loader) # toDO: store this

        # finally, do gtg with the testing set
        with open(os.path.join(feature_test_dir, pkl_name), 'rb') as pkl:
            net_name, labels, features, fnames = pickle.load(pkl)


        print()



if __name__ == '__main__':
    main2()