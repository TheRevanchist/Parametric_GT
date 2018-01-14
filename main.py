import os
import pickle

import feature_extractor as fe
from utils import prepare_dataset, create_dict_nets_and_features, misc, dataset_size, prepare_loader_val
from net import evaluate


def main():
    root = '.'
    current_dataset = 'caltech'
    out_dir_tr = os.path.join(root, 'out', current_dataset, 'feature_data')
    out_dir_test = os.path.join(root, 'out', current_dataset, 'feature_data_test')
    batch_size = 1

    nets_and_features = create_dict_nets_and_features()
    dataset, stats, number_of_classes = misc(root, current_dataset)
    dataset_train_temp, dataset_test_temp = prepare_dataset(dataset)

    for j in xrange(1, 5):
        dataset_train = dataset_train_temp + '_' + str(j)
        dataset_test = dataset_test_temp + '_' + str(j)
        list_of_net_names = ['resnet18', 'resnet152', 'densenet121', 'densenet201']
        dataset_size_train, dataset_size_test = dataset_size(current_dataset)[0], dataset_size(current_dataset)[1]

        for i, net_type in enumerate(list_of_net_names):
            net, feature_size = fe.get_net_info(net_type.split("_")[0], number_of_classes, nets_and_features)

            train_loader = prepare_loader_val(dataset_train, stats, batch_size)
            test_loader = prepare_loader_val(dataset_test, stats, batch_size)

            # if net is densenet
            if net_type[:3] == 'den':
                fc7_features_tr, labels_tr, net_tr, fnames_tr = fe.extract_features_train(net, feature_size, dataset_size_train,
                                                                                 train_loader, dense=1)
                fc7_features_test, labels_test, net_test, fnames_test = fe.extract_features_train(net, feature_size, dataset_size_test,
                                                                                 test_loader, dense=1)
            # if net is resnet
            else:
                fc7_features_tr, labels_tr, net_tr, fnames_tr = fe.extract_features_train(net, feature_size, dataset_size_train,
                                                                                 train_loader, dense=0)
                fc7_features_test, labels_test, net_test, fnames_test = fe.extract_features_train(net, feature_size, dataset_size_test,
                                                                                 train_loader, dense=0)

            # store the name of the net, the dataset on which we are going to use it, and the testing accuracy
            net_info_tr = [net_type.split("_")[0], labels_tr, fc7_features_tr, fnames_tr]
            with open(os.path.join(out_dir_tr, net_type.split("_")[0] + '_' + str(j) + '.pickle'), 'wb') as f:
                pickle.dump(net_info_tr, f, pickle.HIGHEST_PROTOCOL)

            net_info_test = [net_type.split("_")[0], labels_test, fc7_features_test, fnames_test]
            with open(os.path.join(out_dir_test, net_type.split("_")[0] + '_' + str(j) + '.pickle'), 'wb') as f:
                pickle.dump(net_info_test, f, pickle.HIGHEST_PROTOCOL)


if __name__ == '__main__':
    main()
