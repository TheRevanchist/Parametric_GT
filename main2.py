import os
import pickle
import gtg
import utils
import utils2
import FileDataset


def main2():
    root = '.'
    current_dataset = 'caltech'
    out_dir = os.path.join(root, 'out', current_dataset)
    feature_dir = os.path.join(out_dir, 'feature_data')
    net_dir = os.path.join(out_dir, 'net')
    gtg_labels_dir = os.path.join(out_dir, 'gtg_labels')
    nr_classes = 256

    for pkl_name in os.listdir(feature_dir):
        with open(pkl_name, 'rb') as pkl:
            net_name, labels, features, fnames = pickle.load(pkl)

        W = gtg.sim_mat(features)
        nr_objects = features.shape[0]
        labelled, unlabelled = utils2.create_mapping(nr_objects, 0.1)
        ps = utils2.gen_init_rand_probability(labels, labelled, unlabelled, nr_classes)
        gtg_accuracy, Ps_new = utils2.get_accuracy(W, ps, labels, labelled, unlabelled, len(unlabelled))
        gtg_labels = Ps_new.argmax(axis=1)

        names_folds = os.listdir('Datasets/caltech/train')
        names_folds.sort()
        utils2.gen_gtg_label_file(fnames, names_folds, gtg_labels, gtg_labels_dir)

        dataset = FileDataset.CaltechFileDataset('new_labels')





        max_epochs = 10
        batch_size = 8

        dataset, stats, number_of_classes = misc(user, current_dataset)
        nname, ind = net_name.split('_')
        dataset_train = os.path.join(dataset, 'train_' + ind)
        dataset_test = os.path.join(dataset, 'test')

        nets_and_features = utils.create_dict_nets_and_features()
        net_types = ['resnet18']
        out_dir = os.path.join(os.path.join(os.path.join(user, 'Results'), current_dataset), 'nets')

        for net_type in net_types:
            inception = net_type == 'inception'
            train_loader = prepare_loader_train(dataset_train, stats, batch_size, inception)
            test_loader = prepare_loader_val(dataset_test, stats, batch_size, inception)

            net, feature_size = create_net(number_of_classes, nets_and_features, net_type=net_type)
            criterion = nn.CrossEntropyLoss()
            optimizer = optim.Adam(net.parameters(), lr=1e-4)

            best_net = train(net, net_type, train_loader, test_loader, optimizer, criterion, max_epochs, out_dir)

            net.load_state_dict(torch.load(best_net))
            net_accuracy = evaluate(net, test_loader)
            print('Accuracy: ' + str(net_accuracy))


if __name__ == '__main__':
    main2()