import numpy as np
import sklearn.metrics
import time
from scipy import spatial


def gtg(W, X, L, U, max_iter=100, labels=None):
    iter = 0
    Xbin = X[L, :] > 0.0

    while iter < max_iter:
        q = ((X * np.dot(W[:, U], X[U, :])) + (X * np.dot(W[:, L], Xbin))).sum(axis=1)
        u = (np.dot(W[:, U], X[U, :]) + (np.dot(W[:, L], Xbin)))/q[:, np.newaxis]
        # DO not do in-place multiplication!
        X = X * u

        # checking step-by-step accuracy
        if (not labels is None):
            conf = sklearn.metrics.confusion_matrix(labels[U, :], (X[U, :]).argmax(axis=1))
            acc = float(conf.trace()) / conf.sum()
            # P = X.argmax(axis=1)
            # s_correct = (labels[U] == P).sum()
            # acc = float(s_correct)/len(U)
            # print('Accuracy at iter ' + str(iter) + ': ' + str(acc))
            print('Accuracy at iter ' + str(iter) + ': ' + str(acc))

        iter += 1

    return X


def sim_mat(fc7_feats):
    """
    Given a matrix of features, generate the similarity matrix S and sparsify it.
    :param fc7_feats: the fc7 features
    :return: matrix_S - the sparsified matrix S
    """
    print("Something")
    t = time.time()
    pdist_ = spatial.distance.pdist(fc7_feats)
    print('Created distance matrix' + ' ' + str(time.time() - t) + ' sec')

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