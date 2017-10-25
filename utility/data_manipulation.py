from __future__ import division
import numpy as np


def convert_one_hot(x):

    uniq_num = np.sort(np.unique(x))
    max_num = np.shape(uniq_num)[0]

    one_hot = np.zeros((len(x), max_num ) )

    map = dict( (uniq_num[i] , i) for i in xrange(len(uniq_num)) )

    for i in xrange(len(x)):
        one_hot[i][ map[ x[i] ] ] = 1

    return one_hot


def convert_non_one_hot(x):
    n_classes = np.shape(x)[1]
    return np.dot(x , np.arange(n_classes) ).astype('int');

if __name__ == '__main__':
    print(convert_one_hot([6 , 2 , 0 , 8 , 9 ]))
