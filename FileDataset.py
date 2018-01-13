import os
from skimage import io
from skimage import io

import torch.utils.data


class FileDataset(torch.utils.data.Dataset):
    def __init__(self, fname, root_dir, transform):
        with open(fname, 'r') as f:
            data = list(map(lambda line: line.split(' '), f))
            self.fnames = list(map(lambda item: os.path.join(root_dir, item[0]), data))
            self.labs = list(map(lambda item: item[1], data))
        self.root_dir = root_dir
        self.transform = transform

    def __getitem__(self, idx):
        fname = os.path.join(self.root_dir, self.fnames[idx][0])
        img = io.imread(fname)
        if self.transform:
            img = self.transform(img)
        return img, self.labs[idx], self.fnames[idx]

    def __len__(self):
        return len(self.data)


class CaltechFileDataset(FileDataset):
    def __init__(self, fname, root_dir, transform):
        super(CaltechFileDataset, self).__init__(fname, root_dir, transform)
        self.labs = list(map(lambda item: int(item[:3]), self.labs))