"""
Download data and preprocessing.
"""

import numpy as np
import scipy.io
import os
import shutil

DATA_SET = 'SUN'
DIR = '../../../data_exp/xlsa17/data/' + DATA_SET + '/'


def download_data():
    os.system('wget http://datasets.d2.mpi-inf.mpg.de/xian/xlsa17.zip')
    os.system('unzip xlsa17.zip')
    os.remove('xlsa17.zip')


def prepro_data(dataset_name):
    print dataset_name
    os.makedirs(dataset_name)
    path = 'xlsa17/data/'
    mat_feat = scipy.io.loadmat(path + '' + dataset_name + '/res101.mat')
    features = mat_feat['features'].T
    labels = mat_feat['labels']
    mat = scipy.io.loadmat(path + '' + dataset_name + '/att_splits.mat')
    attributes = mat['att'].T
    split_name = ['trainval', 'test_seen', 'test_unseen', 'train', 'val']
    for name in split_name:
        print name
        locs = mat[name + '_loc']
        features_temp = np.zeros((locs.shape[0], features.shape[1]))
        labels_temp = np.zeros((locs.shape[0], np.amax(labels)))
        attributes_temp = np.zeros((locs.shape[0], attributes.shape[1]))
        for i, loc in enumerate(locs):
            features_temp[i] = features[loc - 1]
            labels_temp[i, labels[loc - 1] - 1] = 1
            attributes_temp[i] = attributes[labels[loc - 1] - 1]
        np.save(dataset_name + '/' + DATA_SET + '_' + name + '_features', features_temp)
        np.save(dataset_name + '/' + DATA_SET + '_' + name + '_labels', labels_temp)
        np.save(dataset_name + '/' + DATA_SET + '_' + name + '_attributes', attributes_temp)
    print "======="


download_data()

data_set = ['APY', 'AWA1', 'AWA2', 'CUB', 'SUN']

for name in data_set:
    prepro_data(name)

shutil.rmtree('xlsa17')
