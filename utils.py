import os

import torch
import torchvision
import torch.nn as nn
import torch.utils.data

from ImageFolder2 import ImageFolder2, CaltechImageFolder


datasets = {'imagenet': {'src': 'Datasets/imagenet',
                         'stats': (.485, .456, .406, .229, .224, .225),
                         'nr_classes': 1000,
                         'size': {'train': 50000,
                                  'test': 1}},
            'caltech': {'src': 'Datasets/caltech',
                        'stats': (.517, .5015, .4736, .315, .3111, .324),
                        'nr_classes': 256,
                        'size': {'train': 20730,
                                 'test': 9051}},
            'sun': {'src': 'Datasets/sun'},
                    'stats': (.472749174938, .461143867394, .432053035945, .265962445818, .263875783693, .289179359433),
                    'nr_classes': 100,
                    'size': {'train': 4994,
                             'test': 1}}
nets = {
    'resnet18': torchvision.models.resnet18,
    'resnet34': torchvision.models.resnet34,
    'resnet50': torchvision.models.resnet50,
    'resnet101': torchvision.models.resnet101,
    'resnet152': torchvision.models.resnet152,
    'densenet121': torchvision.models.densenet121,
    'densenet161': torchvision.models.densenet161,
    'densenet169': torchvision.models.densenet169,
    'densenet201': torchvision.models.densenet201
}


def finetune_model(net, nr_classes):
    model = nets[net](pretrained=True)
    layers = list(model.children())
    layers[-1] = nn.Linear(layers[-1].in_features, nr_classes)

    return nn.Sequential(*layers).cuda()


def create_net(number_of_classes, nets_and_features, net_type='resnet152'):
    net, feature_size = nets_and_features[net_type][0](pretrained=True), nets_and_features[net_type][1]
    if net_type[:3] == 'den':
        net.classifier = nn.Linear(feature_size, number_of_classes)
    else:
        net.fc = nn.Linear(feature_size, number_of_classes)
    net = net.cuda()
    return net, feature_size


def prepare_train_loader(dataset, stats, batch_size):
    transform = torchvision.transforms.Compose([
        torchvision.transforms.RandomSizedCrop(224),
        torchvision.transforms.RandomHorizontalFlip(),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(mean=(stats[0], stats[1], stats[2]),
                                         std=(stats[3], stats[4], stats[5]))
    ])

    train = ImageFolder2(dataset, transform)
    train_loader = torch.utils.data.DataLoader(train, batch_size=batch_size, shuffle=True, num_workers=12, pin_memory=True)
    return train_loader


def prepare_val_loader(dataset, stats, batch_size):
    transform = torchvision.transforms.Compose([
        torchvision.transforms.Scale(256),
        torchvision.transforms.CenterCrop(224),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(mean=(stats[0], stats[1], stats[2]),
                                         std=(stats[3], stats[4], stats[5]))
    ])

    val = CaltechImageFolder(dataset, transform)
    val_loader = torch.utils.data.DataLoader(val, batch_size=batch_size, shuffle=False, num_workers=12, pin_memory=True)
    return val_loader
