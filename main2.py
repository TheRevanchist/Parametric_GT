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
import numpy as np
from sklearn import svm



def main2():

    # open the file we have to fill
    results = 'results-1-indoors.txt'
    with open(results, 'w') as file:
        file.write("Net name " + " trial ind " + "gtg " + "svm " + "ann" + "\n")

    root = '.'
    current_dataset = 'indoors'
    out_dir = os.path.join(root, 'out', current_dataset)
    feature_dir = os.path.join(out_dir, 'feature_data')
    feature_test_dir = os.path.join(out_dir, 'feature_data_test')
    svm_labels_dir = os.path.join(out_dir, 'svm_labels')
    net_dir = os.path.join(out_dir, 'nets')
    nets_dir_test = os.path.join(out_dir, 'nets_test')
    gtg_labels_dir = os.path.join(out_dir, 'gtg_labels')
    only_labelled = os.path.join(out_dir, 'only_labelled')
    nr_classes = 256
    nets_and_features = create_dict_nets_and_features()

    for pkl_name in os.listdir(feature_dir):
        with open(os.path.join(feature_dir, pkl_name), 'rb') as pkl:
            net_name, labels, features, fnames = pickle.load(pkl)

        W = gtg.sim_mat(features)
        nr_objects = features.shape[0]
        labelled, unlabelled = utils2.create_mapping2(labels, 0.02)
        ps = utils2.gen_init_rand_probability(labels, labelled, unlabelled, nr_classes)
        gtg_accuracy, Ps_new = utils2.get_accuracy(W, ps, labels, labelled, unlabelled, len(unlabelled))
        gtg_labels = Ps_new.argmax(axis=1)

        nname, ind = pkl_name.split('_')

        names_folds = os.listdir('Datasets/indoors/train_' + str(ind[0]))
        names_folds.sort()
        gtg_label_file = os.path.join(gtg_labels_dir, nname + '.txt')
        utils2.gen_gtg_label_file(fnames, names_folds, gtg_labels, gtg_label_file)

        # generate the new dataset
        gen_gtg_dataset('indoors/train_' + str(ind[0]), gtg_label_file, ind[0])

        stats = (.485, .456, .406, .229, .224, .225)

        del W

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

        trained_net = train(net, nname, train_loader, test_loader, optimizer, criterion, max_epochs, net_dir, ind[0])

        net.load_state_dict(torch.load(trained_net))
        net_accuracy_gtg = evaluate(net, test_loader)
        print('Accuracy: ' + str(net_accuracy_gtg))

        # do the same thing but with a linear SVM
        svm_linear_classifier = svm.LinearSVC()
        svm_linear_classifier.fit(features[labelled,:], labels[labelled])
        labels_svm = svm_linear_classifier.predict(features[unlabelled])

        labels_svm = labels_svm.astype(int)
        gtg_labels[unlabelled] = labels_svm

        svm_label_file = os.path.join(svm_labels_dir, nname + '.txt')
        utils2.gen_gtg_label_file(fnames, names_folds, gtg_labels, svm_label_file)
        gen_gtg_dataset('indoors/train_' + str(ind[0]), svm_label_file, ind[0], 'train_labelled_svm')

        dataset_train = os.path.join(dataset, 'train_labelled_svm_' + ind[0])

        train_loader = prepare_loader_train(dataset_train, stats, batch_size)
        test_loader = prepare_loader_val(dataset_test, stats, batch_size)

        net, feature_size = create_net(nr_classes, nets_and_features, net_type=nname)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(net.parameters(), lr=1e-4)

        trained_net = train(net, nname, train_loader, test_loader, optimizer, criterion, max_epochs, net_dir, ind[0])

        net.load_state_dict(torch.load(trained_net))
        net_accuracy_svm = evaluate(net, test_loader)
        print('Accuracy: ' + str(net_accuracy_svm))


        # now check the accuracy of the net trained only in the labelled set
        label_file = os.path.join(only_labelled, nname + '.txt')
        utils2.only_labelled_file(fnames, labelled, label_file)
        gen_labelled_dataset('indoors/train_' + str(ind[0]), label_file, ind[0])

        dataset_train = os.path.join(dataset, 'train_only_labelled_' + ind[0])

        train_loader = prepare_loader_train(dataset_train, stats, batch_size)
        test_loader = prepare_loader_val(dataset_test, stats, batch_size)

        net, feature_size = create_net(nr_classes, nets_and_features, net_type=nname)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(net.parameters(), lr=1e-4)

        trained_net = train(net, nname, train_loader, test_loader, optimizer, criterion, max_epochs, nets_dir_test, ind[0])

        net.load_state_dict(torch.load(trained_net))
        net_accuracy = evaluate(net, test_loader)


        # # finally, do gtg with the testing set
        # with open(os.path.join(feature_test_dir, pkl_name), 'rb') as pkl:k
        #     net_name_test, labels_test, features_test, fnames_test = pickle.load(pkl)
        #
        # features_combined = np.vstack((features[labelled,:], features_test))
        # labels_combined = np.vstack((labels[labelled], labels_test))
        # W = gtg.sim_mat(features_combined)
        # labelled = np.arange(features[labelled,:].shape[0])
        # unlabelled = np.arange(features[labelled,:].shape[0], features_combined.shape[0])
        #
        # ps = utils2.gen_init_rand_probability(labels_combined, labelled, unlabelled, nr_classes)
        # gtg_accuracy_test, Ps_new = utils2.get_accuracy(W, ps, labels_combined, labelled, unlabelled, len(unlabelled))

        with open(results, 'a') as file:
            file.write(nname + "   " + ind[0] + "   " + str(net_accuracy_gtg) + "   " + str(net_accuracy_svm) + "   " + str(net_accuracy) + "\n")

        print()


if __name__ == '__main__':
    main2()