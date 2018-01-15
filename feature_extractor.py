import torch
import torch.utils.data
import torch.nn as nn
from net import evaluate
import torch.nn.functional as F
import utils


def get_net_info(net_processed_name, number_of_classes, nets_and_features):
    net, feature_size = utils.create_net(number_of_classes, nets_and_features, net_processed_name)
    return net, feature_size


def one_hot(labels, number_of_classes):
    len_ = 1 if isinstance(labels, int) else len(labels)
    label_one_hot = torch.zeros(len_, number_of_classes)
    label_one_hot[list(xrange(len_)), labels] = 1.
    return label_one_hot


def extract_features_train(net, dataset_size, train_loader, dense=False):
    net.eval()
    layers = list(net.children())
    net = nn.Sequential(*layers[:-1])

    features = torch.zeros(dataset_size, layers[-1].in_features)
    labels_ = torch.zeros(dataset_size, 1)
    names_of_files = []

    for k, data in enumerate(train_loader, 0):
        inputs, labels, path = data
        inputs, labels = torch.autograd.Variable(inputs).cuda(), torch.autograd.Variable(labels).cuda()
        outputs = net(inputs)
        if dense:
            outputs = torch.squeeze(F.avg_pool2d(F.relu(outputs), 7))
        else:
            outputs = F.relu(outputs)
        features[k, :] = outputs.data
        labels_[k, :] = labels.data
        names_of_files.append(path)

    fc7_features = features.numpy()
    labels = labels_.numpy()
    return fc7_features, labels, net, names_of_files

