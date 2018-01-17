import numpy as np
import sklearn
import gtg
import os
from math import log
import random


def one_hot(labels, nr_classes):
    labells = labels[:, 0]
    labells = labells.astype(int)
    label_one_hot = np.zeros((labells.size, nr_classes))
    label_one_hot[np.arange(labells.size), labells] = 1
    return label_one_hot


def create_mapping(nr_objects, percentage_labels):
    mapping = np.arange(nr_objects)
    np.random.shuffle(mapping)
    nr_labelled = int(percentage_labels * nr_objects)
    labelled = mapping[:nr_labelled]
    unlabelled = mapping[nr_labelled:]
    return np.sort(labelled), np.sort(unlabelled)


def create_mapping2(labels, percentage_labels):
    nr_classes = int(labels.max() + 1)

    labelled, unlabelled = [], []
    for n_class in xrange(nr_classes):
        class_labels = list(np.where(labels == n_class)[0])
        split = int(percentage_labels * len(class_labels))
        random.shuffle(class_labels)
        labelled += class_labels[:split]
        unlabelled += class_labels[split:]
    return np.array(labelled), np.array(unlabelled)


def gen_init_rand_probability(labels, labelled, unlabelled, nr_classes):
    labels_one_hot = np.zeros((labels.shape[0], nr_classes))
    for element in labelled:
        labels_one_hot[element, int(labels[element])] = 1.0
    for element in unlabelled:
        labels_one_hot[element, :] = np.full((1, nr_classes), 1.0/nr_classes)
    return labels_one_hot


def gen_init_probability(W, labels, labelled, unlabelled):
    """
    :param W: similarity matrix to generate the labels for the unlabelled observations
    :param labels: labels of the already labelled observations
    :return:
    """

    n = W.shape[0]
    k = int(log(n) + 1.)
    # labelled, unlabelled = create_mapping(n, perc_lab)
    W = W[np.ix_(unlabelled, labelled)]

    ps = np.zeros(labels.shape)
    ps[labelled] = labels[labelled]

    max_k_inds = labelled[np.argpartition(W, -k, axis=1)[:, -k:]]
    tmp = np.zeros((unlabelled.shape[0], labels.shape[1]))
    for row in max_k_inds.T:
        tmp += labels[row]
    tmp /= float(k)
    ps[unlabelled] = tmp

    return ps


def get_accuracy(W, softmax_features, labels, labelled, unlabelled, testing_set_size):
    """
    This function computes the accuracy in the testing set
    :param fc7_features: fc7 features for both training and testing set
    :param softmax_features: softmax features for both training and testing set
    :param labels: labels for both training and testing set
    :param accuracy_cnn: the accuracy of cnn (baseline)
    :param testing_set_size: the size of the testing set
    :return: accuracy of our method, accuracy of cnn
    """
    P_new = gtg.gtg(W, softmax_features, labelled, unlabelled, max_iter=25, labels=labels)
    conf = sklearn.metrics.confusion_matrix(labels[unlabelled, :], (P_new[unlabelled, :]).argmax(axis=1))
    return float(conf.trace()) / conf.sum(), P_new


def gen_gtg_label_file(fnames, names_folds, labels_GT, out_fname):
    with open(out_fname, 'w') as file:
        for i in xrange(len(fnames)):
            splitted_name = fnames[i][0].split('/')
            new_name = splitted_name[8] + '/' + splitted_name[9] + ' ' + names_folds[labels_GT[i]] + "\n"
            file.write(new_name)


def only_labelled_file(fnames, labelled, out_fname):
    with open(out_fname, 'w') as file:
        for i in xrange(len(fnames)):
            splitted_name = fnames[i][0].split('/')
            if i in labelled:
                new_name = splitted_name[8] + '/' + splitted_name[9] + ' ' + splitted_name[8] + "\n"
                file.write(new_name)


def unit_test():
    """
    unit_test for gen_init_probability function
    :return:
    """
    np.random.seed(314)
    # unlab = 0, 1, 3. lab = 2, 4, 5
    W = np.array([[5, 3, (8), 4, (9), (1)],
                  [1, 2, (3), 4, (7), (9)],
                  [7, 1,  2 , 8,  4 ,  3 ],
                  [9, 7, (4), 3, (2), (1)],
                  [5, 7,  4 , 2,  8 ,  6 ],
                  [6, 4,  5 , 3,  1 ,  2 ]])

    labels = np.array([[0, 1, 0, 0, 0],
                       [1, 0, 0, 0, 0],
                       [0, 0, 1, 0, 0],
                       [0, 0, 0, 1, 0],
                       [0, 0, 0, 0, 1],
                       [0, 1, 0, 0, 0]])

    res = np.array([[0, 0  , 0.5, 0, 0.5],
                    [0, 0.5, 0  , 0, 0.5],
                    [0, 0  , 1  , 0, 0  ],
                    [0, 0  , 0.5, 0, 0.5],
                    [0, 0  , 0  , 0, 1  ],
                    [0, 1  , 0  , 0, 0  ]])
    print(gen_init_probability(W, labels, 0.5))
