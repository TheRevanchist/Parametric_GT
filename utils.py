import torch
import torchvision

import os
import torch.nn as nn
from ImageFolder2 import ImageFolder3


def misc(user, current_dataset):
    datasets = {'imagenet': os.path.join(user, 'Datasets/imagenet'),
                'caltech': os.path.join(user, 'Datasets/caltech'),
                'sun': os.path.join(user, 'Datasets/sun')}
    stats_datasets = {'imagenet': (.485, .456, .406, .229, .224, .225),
             'caltech': (.517, .5015, .4736, .315, .3111, .324),
             'sun': (.472749174938, .461143867394, .432053035945, .265962445818, .263875783693, .289179359433)}
    num_classes = {'imagenet': 1000, 'caltech': 256, 'sun': 100}
    dataset = datasets[current_dataset]
    stats = stats_datasets[current_dataset]
    number_of_classes = num_classes[current_dataset]
    return dataset, stats, number_of_classes


def dataset_size(dataset):
    sizes = {'imagenet': (50000,1),
             'caltech': (20730, 9051),
             'sun': (4994, 1)}
    return sizes[dataset]


def prepare_dataset(dataset):
    dataset_train = os.path.join(dataset, 'train')
    dataset_test = os.path.join(dataset, 'test')
    return dataset_train, dataset_test


def create_dict_nets_and_features():
    nets_and_features = {
        'resnet18': (torchvision.models.resnet18, 512),
        'resnet34': (torchvision.models.resnet34, 512),
        'resnet50': (torchvision.models.resnet50, 2048),
        'resnet101': (torchvision.models.resnet101, 2048),
        'resnet152': (torchvision.models.resnet152, 2048),
        'densenet121': (torchvision.models.densenet121, 1024),
        'densenet161': (torchvision.models.densenet161, 2208),
        'densenet169': (torchvision.models.densenet169, 1664),
        'densenet201': (torchvision.models.densenet201, 1920),
        'inception': (torchvision.models.inception_v3, 2048)
    }
    return nets_and_features


def create_net(number_of_classes, nets_and_features, net_type='resnet152'):
    net, feature_size = nets_and_features[net_type][0](pretrained=True), nets_and_features[net_type][1]
    if net_type[:3] == 'den':
        net.classifier = nn.Linear(feature_size, number_of_classes)
    else:
        net.fc = nn.Linear(feature_size, number_of_classes)
    net = net.cuda()
    return net, feature_size


def prepare_loader_train(dataset, stats, batch_size):
    transform = torchvision.transforms.Compose([
        torchvision.transforms.RandomSizedCrop(224),
        torchvision.transforms.RandomHorizontalFlip(),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(mean=(stats[0], stats[1], stats[2]),
                                         std=(stats[3], stats[4], stats[5]))
    ])
    train = ImageFolder3(dataset, transform)
    train_loader = torch.utils.data.DataLoader(train, batch_size=batch_size, shuffle=True, num_workers=12, pin_memory=True)
    return train_loader


def prepare_loader_val(dataset, stats, batch_size):
    transform = torchvision.transforms.Compose([
        torchvision.transforms.Scale(256),
        torchvision.transforms.CenterCrop(224),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(mean=(stats[0], stats[1], stats[2]),
                                         std=(stats[3], stats[4], stats[5]))
    ])
    val = ImageFolder3(dataset, transform)
    val_loader = torch.utils.data.DataLoader(val, batch_size=batch_size, shuffle=False, num_workers=12, pin_memory=True)
    return val_loader


