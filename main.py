import os
import pickle

import feature_extractor as fe
from utils import datasets, nets
from utils import prepare_val_loader, finetune_model
from net import evaluate


def main():
    root = '.'
    dataset_name = 'caltech'
    net_models = ['resnet18', 'resnet152', 'densenet121', 'densenet201']
    num_exps = 5
    batch_size = 1
    out_dir = os.path.join(root, 'out', dataset_name, 'feature_data')

    dataset = datasets[dataset_name]
    src, stats, nr_classes, size = dataset['src'], dataset['stats'], dataset['nr_classes'], dataset['size']

    for i in xrange(1, num_exps):
        for set_ in ('train', 'test'):
            set_dir = os.path.join(src, set_ + '_' + str(i))
            set_size = size[set_]

            for net_model in net_models:
                print(nr_classes, net_model)
                ft_model = finetune_model(net_model, nr_classes)
                set_loader = prepare_val_loader(set_dir, stats, batch_size)

                fc7_features_tr, labels, net, fnames = fe.extract_features_train(ft_model, set_size,
                                                                             set_loader, dense='densenet' in net_model)

                # store the name of the net, the dataset on which we are going to use it, and the testing accuracy
                net_info = [net_model, labels, fc7_features_tr, fnames]

                with open(os.path.join(out_dir + set_, net_model + '_' + str(i) + '.pickle'), 'wb') as f:
                    pickle.dump(net_info, f, pickle.HIGHEST_PROTOCOL)


if __name__ == '__main__':
    main()
