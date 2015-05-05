import numpy as np
import scipy
# Import datasets, classifiers and performance metrics
from sklearn import datasets, svm, metrics, preprocessing
import matplotlib.pyplot as plt
import math
from random import randint
import os


def mean_squared(p1, p2):
    assert p1.shape == p2.shape
    return 0.5 * np.sum(np.square(p1 - p2))




class KM(object):
    images = None
    centers = None
    labels = None
    distance = None
    num_points = None

    def __init__(self, images, preprocess=preprocessing.scale, num_clusters=5, distance=mean_squared):
        self.distance = distance
        self.k = num_clusters
        self.centers = [np.random.rand(1, 784) for _ in range(num_clusters)]
        self.labels = np.empty((60000, 1))
        self.images = images / 255.0
        self.num_points = self.images.shape[1]
        print 'initialization done, image shape: {0}'.format(self.images.shape)

    def train(self, iterations=25):
        plt.ion()
        plt.show()
        for iteration in xrange(iterations):
            total_loss = 0
            for i in xrange(self.images.shape[0]):
                img = self.images[i].reshape(1, 784)
                scores = np.array([mean_squared(center, img) for center in self.centers])
                self.labels[i] = np.argmin(scores)
                total_loss += scores[np.argmin(scores)]
            self.centers = [self.recompute_center(k, center) for center, k in zip(self.centers, xrange(self.k))]
            f, axarr = plt.subplots(4, 4)
            for center, c in zip(self.centers, xrange(len(self.centers))):
                axarr[c / 4, c % 4].imshow(center.reshape(28, 28), cmap='Greys')
            print 'Total loss is {0} for iteration {1}'.format(total_loss, iteration)
            plt.draw()

    def recompute_center(self, k, center):
        mean = np.zeros((1, 784))
        total = 0
        for label, i in zip(self.labels, xrange(len(self.labels))):
            if label == k:
                image = self.images[i]
                # print 'image shape: {0}'.format(image.shape)
                mean += image
                total += 1
        if total == 0:
            return center
        return mean / float(total)


class Cluster(object):
    center = None
    data_points = None
    distance = None

    def __init__(self, distance=mean_squared, random=True):
        self.center = np.random.rand(1, 784)
        self.data_points = np.array((0, 784))
        print self.data_points.shape
        self.distance = distance

    def recompute_center(self):
        self.center = np.sum(self.data_points, axis=0).reshape(-1, 1)

    def score(self, point):
        return self.distance(self.center, point)

    def visualize(self):
        return self.center.reshape(28, 28)

    def add(self, point):
        print self.data_points.shape
        self.data_points = np.concatenate((self.data_points, point))
