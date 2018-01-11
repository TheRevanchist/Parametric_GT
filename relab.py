import time
import numpy as np
from scipy import spatial
import pickle


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


features = data[4]
# W = sim_mat(features)

nr_objects = features.shape[0]

print()

def create_mapping(nr_objects, percentage_labels):
    mapping = np.arange(nr_objects)
    np.random.shuffle(mapping)
    nr_labelled = int(percentage_labels * nr_objects)
    labelled = mapping[:nr_labelled]
    unlabelled = mapping[nr_labelled:]
    return np.sort(labelled), np.sort(unlabelled)

labelled, unlabelled = create_mapping(nr_objects, 0.1)

print()

def gen_init_probability(labels, mapping):
    pass

print()
