import pickle
import os
import torch
import torch.utils.data
import torch.nn as nn
from net import evaluate
import numpy as np
import torch.nn.functional as F
import utils


def get_net_info(net_processed_name, number_of_classes, nets_and_features):
    print(number_of_classes, net_processed_name)
    net, feature_size = utils.create_net(number_of_classes, nets_and_features, net_processed_name)
    return net, feature_size


def one_hot(label, number_of_classes):
    label_one_hot = torch.zeros(1, number_of_classes)
    label_one_hot[0, label] = 1.
    return label_one_hot


def extract_features_train(net, feature_size, dataset_size, train_loader, dense=0):
    net.eval()
    new_classifier = nn.Sequential(*list(net.children())[:-1])
    net = new_classifier

    features = torch.zeros(dataset_size, feature_size)
    labels_ = torch.zeros(dataset_size, 1)

    names_of_files = []

    for k, data in enumerate(train_loader, 0):
        inputs, labels, index = data
        print(index)
        inputs, labels = torch.autograd.Variable(inputs).cuda(), torch.autograd.Variable(labels).cuda()
        outputs = net(inputs)
        if dense:
            outputs = torch.squeeze(F.avg_pool2d(F.relu(outputs), 7))
        else:
            outputs = F.relu(outputs)
        features[k, :] = outputs.data
        labels_[k, :] = labels.data
        names_of_files.append(index)

    fc7_features = features.numpy()
    labels = labels_.numpy()
    return fc7_features, labels, net, names_of_files

