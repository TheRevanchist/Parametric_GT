import os
import pickle

import feature_extractor as fe
from utils import prepare_dataset, create_dict_nets_and_features, misc, dataset_size, prepare_loader_val
from net import evaluate


def main():
    root = 'PycharmProjects/Parametric_GT'
    current_dataset = 'caltech'
    out_dir = os.path.join(root, 'out', current_dataset, 'feature_data')
    batch_size = 1

    nets_and_features = create_dict_nets_and_features()
    dataset, stats, number_of_classes = misc(root, current_dataset)
    dataset_train, dataset_test = prepare_dataset(dataset)

    for j in xrange(5):
        dataset_train += '_' + str(j)
        dataset_test += '_' + str(j)
        list_of_net_names = ['resnet18', 'resnet152']  # , 'densenet121', 'densenet201']
        dataset_size_train = dataset_size(current_dataset)

        for i, net_type in enumerate(list_of_net_names):
            net, feature_size = fe.get_net_info(net_type.split("_")[0], number_of_classes, nets_and_features)

            train_loader = prepare_loader_val(dataset_train, stats, batch_size)
            # if net is densenet
            if net_type[:3] == 'den':
                fc7_features_tr, labels, net, fnames = fe.extract_features_train(net, feature_size, dataset_size_train,
                                                                                 train_loader, dense=1)
            # if net is resnet
            else:
                fc7_features_tr, labels, net, fnames = fe.extract_features_train(net, feature_size, dataset_size_train,
                                                                                 train_loader, dense=0)

            # store the name of the net, the dataset on which we are going to use it, and the testing accuracy
            net_info = [net_type.split("_")[0], labels, fc7_features_tr, fnames]
            with open(os.path.join(out_dir, net_type.split("_")[0] + '_' + str(j) + '.pickle'), 'wb') as f:
                pickle.dump(net_info, f, pickle.HIGHEST_PROTOCOL)


if __name__ == '__main__':
    main()
