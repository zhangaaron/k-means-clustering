import numpy as np
import scipy
# Import datasets, classifiers and performance metrics
from sklearn import datasets, svm, metrics, preprocessing
import matplotlib.pyplot as plt
import math
from random import randint
import os


data = io.loadmat("./joke_data/jokes/train.mat")
print data