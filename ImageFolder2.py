import torchvision


class ImageFolder3(torchvision.datasets.ImageFolder):
    def __getitem__(self, index):
        img, target = super(ImageFolder3, self).__getitem__(index)
        path = self.imgs[index][0]
        return img, target, path
