# Correlation Discriminant Analysis
# Author:        Thomas Greaney <t9reaney@gmail.com>
# Created:       7th July 2023
# Last Modified: 7th July 2023

import copy
import numpy as np
import scipy.stats as stats
from sklearn.utils.multiclass import unique_labels


def getCorrelationVectors(X, Y):
    """
    :param X: array of shape (n_samples, n_features)
              Training Data
    :param Y: array of shape (n_samples, n_classes)
              one-hot encoded array for each target class
    :return:  array of shape (n_features, n_classes)
              correlations between each feature and each target class
    """

    correlationVectors = []
    for classLabel in Y:
        correlationVector = getCorrelationVector(X, classLabel)
        correlationVectors.append(correlationVector)

    return correlationVectors


def getCorrelationVector(X, y):
    """
    :param X: array of shape (n_samples, n_features)
              Training Data
    :param y: array of shape (n_samples)
              one-hot encoded array for target class
    :return:  array of shape (n_features)
              correlation between each feature and the target class
    """
    variableCorrelations = []
    # get transverse array
    npX = np.array(X).T

    for variable in npX:
        # calculate pearson correlation
        correlation = stats.pearsonr(variable, y)[0]
        variableCorrelations.append(correlation)
    return variableCorrelations


def oneHotEncodedLabels(y):
    """
    :param y: array of shape (n_samples)
              labels of classe for each sample
    :return:  array of shape (n_samples, n_classes)
              one hot encoded array for each class target
    """
    labels = unique_labels(y)
    encodedLabels = []
    for label in labels:
        oneHotEncodedLabel = []
        for sample in y:
            if sample == label:
                oneHotEncodedLabel.append(1)
            else:
                oneHotEncodedLabel.append(0)
        encodedLabels.append(oneHotEncodedLabel)

    return encodedLabels


def getAccuracy(predictedVals, targetVals):
    if len(predictedVals != len(targetVals)):
        return -1

    numCorrectPredictions = 0

    for i in range(0, len(targetVals)):
        if predictedVals[i] == targetVals[i]:
            numCorrectPredictions = numCorrectPredictions + 1

    accuracy = numCorrectPredictions / len(targetVals)

    return accuracy


def clipVectors(vectors, X, Y):
    """
    Removes values from a vector inside a certain range to optimize performance

    :param X:
    :param vectors:
    :return:
    """
    return vectors


class CorrelationDiscriminantAnalysis:

    def __init__(
            self,
            n_components=None
    ):
        # number of features to use when training
        self.n_components = n_components
        # correlation vectors to use for unseen data
        self.correlationVectors = None
        # class labels
        self.classes = None
        # number of classes in prediction
        self.numClasses = None
        # clipping value for correlation vector to remove noise
        self.clippingRange = 1

    def fit(self, x, y, enableClipping=False):
        """
        Fit the Correlation Discriminant Analysis model.

        :param x:              array-like of shape (n_samples, n_features)
                               Training data.
                               Assumes data in X is of normal distribution
                               requires categorical data to be one-hot encoded
                               requires no minority classes in X

        :param y:              array-like of shape (n_samples,)
                               Target values.

        :param enableClipping: boolean
                               determines whether correlation vectors are transformed for optimisation

        :return:               self : object
                               Fitted estimator.
        """

        encodedLabels = oneHotEncodedLabels(y)
        self.correlationVectors = getCorrelationVectors(x, encodedLabels)
        self.classes = unique_labels(y)
        self.numClasses = len(self.classes)

        if enableClipping:
            self.correlationVectors = clipVectors(self.correlationVectors, x)

        return self

    def predict(self, x):
        """
        Predict unseen data using the Correlation model.

        :param x: array-like or sparse matrix, shape (n_samples, n_features)
                  Unseen data
        :return:  C : array, shape (n_samples,)
                  Returns predicted values.
        """

        if self.correlationVectors is None or self.classes is None:
            print("Error, model has not been fitted to training data")
            return None

        predictions = self.__correlationPredict(x, self.correlationVectors)

        return predictions

    def __correlationPredict(self, x, correlationVectors):
        """
        Predict unseen data using arbitrary correlation vectors.

        :param x:                   array-like or sparse matrix, shape (n_samples, n_features)
                                    Unseen data
        :param correlationVectors:  array of shape (n_features, n_classes)
                                    correlations between each feature and each target class
        :return:                    C : array, shape (n_samples,)
                                    Returns predicted values.
        """
        predictions = []

        for sample in x:
            maxStrength = float('-inf')
            closestClass = -1
            for i in range(0, self.numClasses):
                classCorrelationVector = correlationVectors[i]
                classStrength = np.sum(np.dot(sample, classCorrelationVector))
                if classStrength > maxStrength:
                    maxStrength = classStrength
                    closestClass = i

            predictions.append(self.classes[closestClass])

        return predictions

    def __calcClippedAccuracy(self, x, y, c):
        """

        :param x: training data
        :param y: encoded labels
        :param c: clipping range
        :return:
        """

        clippedVectors = copy.deepcopy(self.correlationVectors)

        

        return 0
