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

            os.makedirs(train_folder)
            os.makedirs(test_folder)

            files = os.listdir(source_folder)
            random.shuffle(files)

            split = int(tr_perc * len(files))

            for file in files[:split]:
                shutil.copy(os.path.join(source_folder, file), train_folder)

            for file in files[split:]:
                shutil.copy(os.path.join(source_folder, file), test_folder)


def gen_gtg_dataset(dataset_name, data_fname, lab_perc):
    source = os.path.join(root, dataset_name)

    with open(data_fname, 'r') as f:
        for line in f:
            fname, lab = line.split(sep=' ')
            dst_pname = os.path.join(root, 'train_gtg_' + str(lab_perc), lab)
            os.makedirs(dst_pname)
            shutil.copy(os.path.join(source, fname), dst_pname)


if __name__ == '__main__':
    # split('caltech', 0.7, n=1)
    gen_gtg_dataset('caltech', '<insert name of lab file here>', 0.1)
