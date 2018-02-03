import os
import shutil
import random

root = 'Datasets'


def split(dataset_name, tr_perc, n=1):
    source = os.path.join(root, dataset_name)

    for i in range(n):
        for folder in os.listdir(source):
            source_folder = os.path.join(source, folder)
            train_folder = os.path.join(root, 'indoors', 'train_' + str(i), folder)
            test_folder = os.path.join(root, 'indoors', 'test_' + str(i), folder)

            try:
                os.makedirs(train_folder)
                os.makedirs(test_folder)
            except IOError:
                pass

            files = os.listdir(source_folder)
            random.shuffle(files)

            split = int(tr_perc * len(files))

            for file in files[:split]:
                shutil.copy(os.path.join(source_folder, file), train_folder)

            for file in files[split:]:
                shutil.copy(os.path.join(source_folder, file), test_folder)


def gen_gtg_dataset(dataset_name, data_fname, ind, out_folder='train_labelled'):
    source = os.path.join(root, dataset_name)
    counter = 0
    with open(data_fname, 'r') as f:
        for line in f:
            try:
                fname, lab = line.split(' ')
                lab = lab[:-1]
                dst_pname = os.path.join(root, 'indoors', out_folder + '_' + str(ind), lab)
                try:
                    os.makedirs(dst_pname)
                except OSError:
                    pass
                shutil.copy(os.path.join(source, fname), dst_pname)
            except ValueError:
                continue

    print(counter)
    print("\n\n")


def gen_labelled_dataset(dataset_name, data_fname, ind):
    source = os.path.join(root, dataset_name)
    counter = 0
    with open(data_fname, 'r') as f:
        for line in f:
            try:
                fname, lab = line.split(' ')
            except ValueError:
                counter += 1
            lab = lab[:-1]
            dst_pname = os.path.join(root, 'indoors', 'train_only_labelled' + '_' + str(ind), lab)
            try:
                os.makedirs(dst_pname)
            except OSError:
                pass
            shutil.copy(os.path.join(source, fname), dst_pname)


if __name__ == '__main__':
    split('indoors/indoors_src', 0.7, n=1)
    random.seed(2718)
    # gen_gtg_dataset('caltech/train', 'only_labelled.txt', 0.1)
