import os
import shutil
import random

root = 'Datasets'


def make_source_file(dataset_name):
    source = os.path.expanduser(dataset_name)
    if not os._exists('dataset_files'):
        os.makedirs('dataset_files')

    with open('dataset_files/source_file.txt', 'w') as f:
        for class_dir in os.listdir(source):
            for fname in os.listdir(os.path.join(source, class_dir)):
                f.write(os.path.join(class_dir, fname) + '\n')


def split_source_file(source_file, tr_perc=0.7, n=1):
    with open(source_file, 'r') as f:
        f = list(f)
        split = int(tr_perc * len(f))
        for i in xrange(n):
            random.shuffle(f)
            with open('train_' + str(i) + '.txt', 'w') as train:
                for line in f[:split]:
                    train.write(line)

            with open('test_' + str(i) + '.txt', 'w') as test:
                for line in f[split:]:
                    test.write(line)


def split(dataset_name, tr_perc, n=1):
    source = os.path.join(root, dataset_name)

    for i in range(n):
        for folder in os.listdir(source):
            source_folder = os.path.join(source, folder)
            train_folder = os.path.join(root, 'caltech', 'train_' + str(i), folder)
            test_folder = os.path.join(root, 'caltech', 'test_' + str(i), folder)

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


def gen_gtg_dataset(dataset_name, data_fname, lab_perc):
    source = os.path.join(root, dataset_name)

    with open(data_fname, 'r') as f:
        for line in f:
            fname, lab = line.split(' ')
            lab = lab[:-1]
            dst_pname = os.path.join(root, 'caltech', 'train_labelled' + str(lab_perc), lab)
            try:
                os.makedirs(dst_pname)
            except OSError:
                pass
            shutil.copy(os.path.join(source, fname), dst_pname)


if __name__ == '__main__':
    random.seed(2718)
    # make_source_file('Datasets/caltech/caltech_src')
    split_source_file('dataset_files/source_file.txt', tr_perc=0.7, n=5)
    # split('caltech/caltech_src', 0.7, n=5)
    # gen_gtg_dataset('caltech/train', 'only_labelled.txt', 0.1)
