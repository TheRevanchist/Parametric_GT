import torchvision.datasets
from torchvision.datasets import folder


class ImageFolder2(torchvision.datasets.ImageFolder):
    def __getitem__(self, index):
        img, target = super(ImageFolder2, self).__getitem__(index)
        path = self.imgs[index][0]
        return img, target, path


class CaltechImageFolder(torchvision.datasets.ImageFolder):
    def __init__(self, fnames, transform=None, target_transform=None, loader=folder.default_loader):
        paths = map(lambda item: item.split(' ')[0])
        self.classes = list(map(lambda item: int(item.split(' ')[1][:3]), fnames))
        self.imgs = list(zip(paths, self.classes))

        self.transform = transform
        self.target_transform = target_transform
        self.loader = loader
