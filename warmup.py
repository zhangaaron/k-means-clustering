from scipy import io
# Import datasets, classifiers and performance metrics
from sklearn import datasets, svm, metrics, preprocessing
from sklearn.cross_validation import train_test_split
import matplotlib.pyplot as plt
import time
import numpy as np




def do_stupidity(ratings):
    joke_scores = np.sum(ratings, axis=0)
    answers = None
    num_lines = sum(1 for line in open('./joke_data/validation.txt', 'rb'))
    with open('./joke_data/validation.txt', 'rb') as file:
        answers = np.empty((num_lines, 1))
        for line_num, line in enumerate(file):
            user, joke, validation_answer = map(lambda x: int(x), line.rsplit(','))
            prediction = (1 if joke_scores[joke - 1] >= 0 else 0)
            answers[line_num] = 1.0 if validation_answer == prediction else 0.0
        percentage_correct = np.sum(answers) / float(num_lines)
        print 'total percentage correct = {0}'.format(percentage_correct)


def mean_squared(p1, p2):
    assert p1.shape == p2.shape
    return 0.5 * np.sum(np.square(p1 - p2))

def curried_mean_squared(p1):
    def curry(p2):
        assert p1.shape == p2.shape
        return 0.5 * np.sum(np.square(p1 - p2))
    return curry


class KNN(object):
    k = None
    distance = None
    ratings = None
    cached = {}

    def __init__(self, ratings, k=1000, distance=mean_squared):
        self.k = k
        self.distance = distance
        self.ratings = ratings

    def predict_joke_score(self, user, joke):
        k_nearest = None
        if user in self.cached:
            k_nearest = self.cached[user]
        else:
            k_nearest = self.return_k_nearest_users(user)
            self.cached[user] = k_nearest
        return 1 if k_nearest[joke - 1] >= 0 else 0

    def return_k_nearest_users(self, user):
        user_vector = self.ratings[user - 1]  # users are not zero indexed.
        distances = (self.ratings - user_vector) ** 2
        summed_distance = np.sum(distances, 1)
        sorted_distances = np.argsort(summed_distance)
        indicies = sorted_distances[1: self.k + 1]  # smallest one will be the user itself
        knn_array = np.concatenate([self.ratings[index] for index in indicies]).reshape(len(indicies), -1)
        # print 'KNN SHAPE {0}'.format(knn_array.shape)
        # print 'SUMMED KNN SHAPE {0}'.format(np.sum(knn_array, 0).shape)
        return np.sum(knn_array, 0)

    def do_predictions(self):
        num_lines = sum(1 for line in open('./joke_data/validation.txt', 'rb'))
        with open('./joke_data/validation.txt', 'rb') as file:
            answers = np.empty((num_lines, 1))
            for line_num, line in enumerate(file):
                user, joke, validation_answer = map(lambda x: int(x), line.rsplit(','))
                prediction = self.predict_joke_score(user, joke)
                answers[line_num] = 1.0 if validation_answer == prediction else 0.0
            percentage_correct = np.sum(answers) / float(num_lines)
            print 'percentage correct: {0}'.format(percentage_correct)

data = io.loadmat("./joke_data/joke_train.mat")
ratings = np.nan_to_num(data['train'])
your_dear_nearest_neighbor = KNN(ratings)
your_dear_nearest_neighbor.do_predictions()
