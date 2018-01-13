import pickle
import os

with open('out/caltech/feature_data/resnet18_0.pickle', 'rb') as f:
    data = pickle.load(f)

names = data[4]
for i in xrange(len(names)):
    splitted_name = names[i][0].split('/')
    new_name = splitted_name[8] + '/' + splitted_name[9]
    print(new_name)

