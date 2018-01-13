import time
import numpy as np
from scipy import spatial
from math import floor, log
import pickle
import sklearn.metrics
from gtg import gtg
import os
from pathlib2 import Path

np.random.seed(2718)


def sim_mat(fc7_feats):
    """
    This function creates and sparsifies the matrix S
    :param fc7_feats: the fc7 features
    :return: matrix_S - the sparsified matrix S
    """
    t = time.time()
    pdist_ = spatial.distance.pdist(fc7_feats)
    print('Created distance matrix' + ' ' + str(time.time() - t) + ' sec')

    t = time.time()
    pdist_ = pdist_.astype(np.float32)
    print('Converted to float32' + ' ' + str(time.time() - t) + ' sec')

    t = time.time()
    dist_mat = spatial.distance.squareform(pdist_)
    print('Created square distance matrix' + ' ' + str(time.time() - t) + ' sec')
    del pdist_

    t = time.time()
    sigmas = np.sort(dist_mat, axis=1)[:, 7] + 1e-16
    matrice_prodotti_sigma = np.dot(sigmas[:, np.newaxis], sigmas[np.newaxis, :])
    print('Generated Sigmas' + ' ' + str(time.time() - t) + ' sec')

    t = time.time()
    dist_mat /= -matrice_prodotti_sigma
    print('Computed dists/-sigmas' + ' ' + str(time.time() - t) + ' sec')

    del matrice_prodotti_sigma

    t = time.time()
    W = np.exp(dist_mat, dist_mat)
    # W = np.exp(-(dist_mat / matrice_prodotti_sigma))
    np.fill_diagonal(W, 0.)

    # sparsify the matrix
    k = int(np.floor(np.log2(fc7_feats.shape[0])) + 1)
    n = W.shape[0]
    print('Created inplace similarity matrix' + ' ' + str(time.time() - t) + ' sec')

    t = time.time()
    for x in W:
        x[np.argpartition(x, n - k)[:(n - k)]] = 0.0

    print('Sparsify the matrix' + ' ' + str(time.time() - t) + ' sec')

    t = time.time()
    # matrix_S = np.zeros((n, n))
    m1 = W[np.triu_indices(n, k=1)]
    m2 = W.T[np.triu_indices(n, k=1)]

    W = spatial.distance.squareform(np.maximum(m1, m2))
    print('Symmetrized the similarity matrix' + ' ' + str(time.time() - t) + ' sec')

    return W


with open('Results/caltech/results/resnet18.pickle', 'rb') as f:
    data = pickle.load(f)

labels = data[2]
features = data[3]
names_of_files = data[4]

W = Path('W.pickle')
if W.exists():
    with open('W.pickle', 'rb') as f:
        W = pickle.load(f)
else:
    W = sim_mat(features)
    with open('W.pickle', 'wb') as f:
        pickle.dump(W, f, pickle.HIGHEST_PROTOCOL)

nr_classes = 256  # to be parametrized

nr_objects = features.shape[0]


def create_mapping(nr_objects, percentage_labels):
    mapping = np.arange(nr_objects)
    np.random.shuffle(mapping)
    nr_labelled = int(percentage_labels * nr_objects)
    labelled = mapping[:nr_labelled]
    unlabelled = mapping[nr_labelled:]
    return np.sort(labelled), np.sort(unlabelled)

labelled, unlabelled = create_mapping(nr_objects, 0.1)

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

labells = labels[:, 0]
labells = labells.astype(int)
label_one_hot = np.zeros((labells.size, nr_classes))
label_one_hot[np.arange(labells.size), labells] = 1
# labels_prob_seba = gen_init_probability(W, label_one_hot, labelled, unlabelled)
labels_prob_uni = gen_init_rand_probability(labels, labelled, unlabelled, nr_classes)
print()

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
    P_new = gtg(W, softmax_features, labelled, unlabelled, max_iter=1, labels=labels)
    conf = sklearn.metrics.confusion_matrix(labels[:testing_set_size, :], (P_new[:testing_set_size, :]).argmax(axis=1))
    return float(conf.trace()) / conf.sum(), P_new


acc, P_new = get_accuracy(W, labels_prob_uni, labels, labelled, unlabelled, len(unlabelled))
labels_GT = np.argmax(P_new, axis=1)

names_folds = os.listdir('Datasets/caltech/train')
names_folds.sort()

file = open('new_labels.txt','w')

for i in xrange(len(names_of_files)):
    splitted_name = names_of_files[i][0].split('/')
    new_name = splitted_name[8] + '/' + splitted_name[9] + ' ' + names_folds[labels_GT[i]] + "\n"
    file.write(new_name)
file.close()

file = open('only_labelled.txt', 'w')
# and here we create a similar file just for the labelled data
for i in xrange(len(names_of_files)):
    splitted_name = names_of_files[i][0].split('/')
    if i in labelled:
        new_name = splitted_name[8] + '/' + splitted_name[9] + ' ' + splitted_name[8] + "\n"
        file.write(new_name)
file.close()


