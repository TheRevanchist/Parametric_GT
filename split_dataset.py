import os
import shutil
import random

root = 'Datasets'


def split(dataset_name, tr_perc, n=1):
    source = os.path.join(root, dataset_name)
    folders = os.listdir(source)
    for i in range(n):

        for folder in os.listdir(source):
            source_folder = os.path.join(source, folder)
            train_folder = os.path.join(root, 'train_' + str(n), folder)
            test_folder = os.path.join(root, 'test_' + str(n), folder)

            os.makedirs(os.path.join(root, 'train_' + str(n), folder))
            os.makedirs(os.path.join(root, 'test_' + str(n), folder))

            files = os.listdir(source_folder)
            random.shuffle(files)

            split = int(tr_perc * len(files))

            for file in files[:split]:
                shutil.copy(os.path.join(source_folder, file), train_folder)

            for file in files[split:]:
                shutil.copy(os.path.join(source_folder, file), test_folder)


split('caltech', 0.7, n=1)
