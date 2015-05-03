import numpy as np
from scipy import io
# Import datasets, classifiers and performance metrics
from sklearn import datasets, svm, metrics, preprocessing
from sklearn.cross_validation import train_test_split
import matplotlib.pyplot as plt
import time

def resize_data(data):
    return np.transpose(data.reshape(28 * 28, -1))

def shuffle_and_resize(data):
    labels = data["train_labels"].ravel()
    features = resize_data(data["train_images"])
    assert len(labels) == len(features)
    #consistent shuffling src: http://stackoverflow.com/questions/4601373/better-way-to-shuffle-two-numpy-arrays-in-unison
    rng_state = np.random.get_state()
    np.random.shuffle(labels)
    np.random.set_state(rng_state)
    np.random.shuffle(features)
    return labels, features

def write_to_file(array):
    file = open('kaggle_aaron.csv', 'wb')
    file.write("Id,Category\n")
    for i, value in zip(xrange(1, len(array) + 1), array):
        file.write("%d,%d\n" % (i, value))
    file.close()

data = io.loadmat("./digit-dataset/train.mat")
# test_data = io.load
# test_data = io.loadmat("./test.mat")
# test_feature = resize_data(test_data["test_images"])
labels, features = shuffle_and_resize(data)

training_labels = labels[:55000]
training_features = features[:55000]


t0 = time.time()

N.train()
t1 = time.time()
print 'total time in seconds: {0}'.format((t1-t0)/60.0)
