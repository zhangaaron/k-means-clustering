from scipy import io
# Import datasets, classifiers and performance metrics
from sklearn import datasets, svm, metrics, preprocessing
from sklearn.cross_validation import train_test_split
import matplotlib.pyplot as plt
import time
import numpy as np


"""
dimension U: (24983, 100)
dimension s: (100,)
dimension V: (100, 100)
"""
def PCA(ratings, d=10):
    U, s, V = np.linalg.svd(ratings, False)
    user_matrix = U[:, :d]
    jokes_matrix = V[:d, :]
    print 'user_matrix size: {0}'.format(user_matrix.shape)
    print 'jokes size: {0}'.format(jokes_matrix.shape)
    return user_matrix, jokes_matrix


def do_predictions(ratings):
    user_matrix, jokes_matrix = PCA(ratings)
    MSE = np.sum((np.dot(user_matrix, jokes_matrix) - ratings) ** 2)
    print 'MSE: {0}'.format(MSE)
    num_lines = sum(1 for line in open('./joke_data/validation.txt', 'rb'))
    with open('./joke_data/validation.txt', 'rb') as file:
        answers = np.empty((num_lines, 1))
        for line_num, line in enumerate(file):
            user, joke, validation_answer = map(lambda x: int(x), line.rsplit(','))
            prediction = 1 if np.dot(user_matrix[user - 1], jokes_matrix[:, joke - 1]) >= 0 else 0
            answers[line_num] = 1.0 if validation_answer == prediction else 0.0
        percentage_correct = np.sum(answers) / float(num_lines)
        print 'percentage correct: {0}'.format(percentage_correct)


def nan_mask(matrix):
    matrix = np.abs(matrix)
    matrix = matrix + 1.0  # broadcast this
    matrix = np.nan_to_num(matrix)
    return preprocessing.binarize(matrix)


class PICA(object):
    user_matrix = None
    jokes_matrix = None
    ratings = None
    LAM = 30
    ETA = 0.00001
    nan_mask = None
    ratings = None

    def __init__(self, ratings, d=2):
        self.ratings = np.nan_to_num(ratings)
        assert not np.isnan(self.ratings).any()
        self.user_matrix = np.random.rand(ratings.shape[0], d)
        self.jokes_matrix = np.random.rand(ratings.shape[1], d).T
        self.jokes_matrix = self.jokes_matrix.T
        self.nan_mask = nan_mask(ratings)
        assert not np.isnan(self.nan_mask).any()
        print self.nan_mask

    def do_users_update(self):
        MSE_X_V = np.dot(np.multiply(np.dot(self.user_matrix, self.jokes_matrix.T) - self.ratings, self.nan_mask), self.jokes_matrix)
        #MSE_X_V = np.dot(np.dot(self.user_matrix, self.jokes_matrix.T) - self.ratings, self.jokes_matrix)
        assert np.dot(self.user_matrix, self.jokes_matrix.T).shape == self.ratings.shape
        self.user_matrix += -2 * self.ETA * (MSE_X_V + self.LAM * self.user_matrix)

    def do_jokes_update(self):
        # MSE = np.dot(self.user_matrix, self.jokes_matrix.T) - self.ratings
        # assert not np.isnan(MSE).any()
        MSE = np.multiply(np.dot(self.user_matrix, self.jokes_matrix.T) - self.ratings, self.nan_mask)
        self.jokes_matrix += -2 * self.ETA * (2 * np.dot(self.user_matrix.T, MSE) + 2 * self.LAM * self.jokes_matrix.T).T

    def do_predictions(self):
        MSE = np.sum((np.dot(self.user_matrix, self.jokes_matrix.T) - ratings) ** 2)
        print 'MSE: {0}'.format(MSE)
        num_lines = sum(1 for line in open('./joke_data/validation.txt', 'rb'))
        with open('./joke_data/validation.txt', 'rb') as file:
            answers = np.empty((num_lines, 1))
            for line_num, line in enumerate(file):
                user, joke, validation_answer = map(lambda x: int(x), line.rsplit(','))
                prediction = 1 if np.dot(self.user_matrix[user - 1], self.jokes_matrix.T[:, joke - 1]) >= 0 else 0
                answers[line_num] = 1.0 if validation_answer == prediction else 0.0
            percentage_correct = np.sum(answers) / float(num_lines)
            print 'percentage correct: {0}'.format(percentage_correct)

    def write_test_set(self):
        

    def gradient_descent(self, iterations=10000, debug=True):
        for i in xrange(iterations):
            if i % 500 == 0 and debug:
                print 'iteration # {0}'.format(i)
                self.do_predictions()
            self.do_users_update()
            self.do_jokes_update()

data = io.loadmat("./joke_data/joke_train.mat")
ratings = np.nan_to_num(data['train'])


a = PICA(ratings)
a.gradient_descent()
