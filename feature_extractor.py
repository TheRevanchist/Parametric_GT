import pickle
import os
import torch
import torch.utils.data
import torch.nn as nn
from utils import prepare_dataset, create_dict_nets_and_features, misc, dataset_size, create_net, prepare_loader_val
from net import evaluate
import numpy as np
import torch.nn.functional as F


def main():
    user = os.path.expanduser("~")
    user = os.path.join(user, 'PycharmProjects/Parametric_GT')
    current_dataset = 'caltech'
    out_dir = os.path.join(os.path.join(os.path.join(user, 'Results'), current_dataset), 'results')
    batch_size = 1

    nets_and_features = create_dict_nets_and_features()
    dataset, stats, number_of_classes = misc(user, current_dataset)
    dataset_train, dataset_val, dataset_test = prepare_dataset(dataset)

    nets_folder = os.path.join(os.path.join(os.path.join(user, 'Results'), current_dataset), 'nets')
    inception = 0

    imagenet = current_dataset == 'imagenet'

    list_of_net_names = ['resnet18']

    dataset_size_train, dataset_size_test = dataset_size(current_dataset)[0], dataset_size(current_dataset)[1]

    for i, net_type in enumerate(list_of_net_names):
        net, feature_size = get_net_info(net_type.split("_")[0], number_of_classes, nets_and_features)

        train_loader = prepare_loader_val(dataset_train, stats, batch_size, inception=inception)
        test_loader = prepare_loader_val(dataset_test, stats, batch_size, inception=inception)

        accuracy = evaluate(net, test_loader, net_type)

        softmax_features_train, softmax_labels_train = extract_softmax_train(net, number_of_classes, dataset_size_train,
                                                                             train_loader, number_of_classes)
        # if net is inception
        if net_type == 'inception':
            fc7_features_train, feature_labels_train, net = extract_features_train(net, feature_size, dataset_size_train,
                                                                                   train_loader, dense=0, inception=1)
        # if net is densenet
        elif net_type[:3] == 'den':
            fc7_features_train, feature_labels_train, net = extract_features_train(net, feature_size, dataset_size_train,
                                                                                   train_loader, dense=1, inception=0)
        # if net is resnet
        else:
            fc7_features_train, feature_labels_train, net = extract_features_train(net, feature_size, dataset_size_train,
                                                                                   train_loader, dense=0, inception=0)

        # store the name of the net, the dataset on which we are going to use it, and the testing accuracy
        net_info = [net_type.split("_")[0], accuracy, softmax_features_train, softmax_labels_train, fc7_features_train]
        with open(os.path.join(out_dir, net_type.split("_")[0] + '.pickle'), 'wb') as f:
            pickle.dump(net_info, f, pickle.HIGHEST_PROTOCOL)


def get_net_info(net_processed_name, number_of_classes, nets_and_features):
    print(number_of_classes, net_processed_name)
    net, feature_size = create_net(number_of_classes, nets_and_features, net_processed_name)
    return net, feature_size


def extract_softmax_test(net, classes_number, dataset_size, test_loader, inception=0):
    net.eval()
    features = torch.zeros(dataset_size, classes_number)
    labels_ = torch.zeros(dataset_size, 1)
    m = nn.Softmax()

    for j, data in enumerate(test_loader, 0):
        inputs, labels = data
        inputs, labels = torch.autograd.Variable(inputs).cuda(), torch.autograd.Variable(labels).cuda()
        outputs = net(inputs)
        if inception:
            outputs = outputs[0]
        outputs = m(outputs)
        features[j, :] = outputs.data
        labels_[j, :] = labels.data
    softmax_features = features.numpy()
    labels = labels_.numpy()
    return softmax_features, labels


def extract_softmax_train(net, classes_number, dataset_size, train_loader, number_of_classes):
    net.eval()
    features = torch.zeros(dataset_size, classes_number)
    labels_ = torch.zeros(dataset_size, 1)
    for k, data in enumerate(train_loader, 0):
        inputs, labels = data
        features[k, :] = create_softmax_from_ground_truth(labels[0], number_of_classes)
        labels_[k, :] = labels

    softmax_features = features.numpy()
    labels = labels_.numpy()
    return softmax_features, labels


def create_softmax_from_ground_truth(label, number_of_classes):
    label_one_hot = torch.zeros(1, number_of_classes)
    label_one_hot[0, label] = 1.
    return label_one_hot


def extract_features_test(net, feature_size, dataset_size, test_loader, dense=0, inception=0):
    net.eval()

    features = torch.zeros(dataset_size, feature_size)
    labels_ = torch.zeros(dataset_size, 1)

    for j, data in enumerate(test_loader, 0):
        inputs, labels = data
        inputs, labels = torch.autograd.Variable(inputs).cuda(), torch.autograd.Variable(labels).cuda()
        outputs = net(inputs)
        if inception:
            outputs = outputs[1]
            outputs = F.relu(outputs)
        elif dense:
            outputs = torch.squeeze(F.avg_pool2d(F.relu(outputs), 7))
        else:
            outputs = F.relu(outputs)
        features[j, :] = outputs.data
        labels_[j, :] = labels.data

    fc7_features = features.numpy()
    labels = labels_.numpy()
    return fc7_features, labels


def extract_features_train(net, feature_size, dataset_size, train_loader, dense=0, inception=0):
    net.eval()
    if not inception:
        new_classifier = nn.Sequential(*list(net.children())[:-1])
        net = new_classifier

    features = torch.zeros(dataset_size, feature_size)
    labels_ = torch.zeros(dataset_size, 1)

    for k, data in enumerate(train_loader, 0):
        inputs, labels = data
        inputs, labels = torch.autograd.Variable(inputs).cuda(), torch.autograd.Variable(labels).cuda()
        outputs = net(inputs)
        if inception:
            outputs = outputs[1]
            outputs = F.relu(outputs)
        elif dense:
            outputs = torch.squeeze(F.avg_pool2d(F.relu(outputs), 7))
        else:
            outputs = F.relu(outputs)
        features[k, :] = outputs.data
        labels_[k, :] = labels.data

    fc7_features = features.numpy()
    labels = labels_.numpy()
    return fc7_features, labels, net


if __name__ == "__main__":
    main()
