import time
import numpy as np
from scipy import spatial
from math import log
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
