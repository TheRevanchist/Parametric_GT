import pickle
import os

with open('Results/caltech/results/resnet18.pickle', 'rb') as f:
    data = pickle.load(f)

names_folds = os.listdir('Datasets/caltech/train')
names_folds.sort()

names = data[4]
for i in xrange(len(names)):
    splitted_name = names[i][0].split('/')
    new_name = splitted_name[8] + '/' + splitted_name[9]
    print(new_name)

