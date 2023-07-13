# Correlation Discriminant Analysis
# Author:        Thomas Greaney <t9reaney@gmail.com>
# Created:        7th July 2023
# Last Modified: 13th July 2023
from sklearn.utils.multiclass import unique_labels
import copy
import math
import numpy as np
import scipy.stats as stats


def getCorrelationVectors(x, y):
    """
    :param x: array of shape (n_samples, n_features)
              Training Data
    :param y: array of shape (n_samples, n_classes)
              one-hot encoded array for each target class
    :return:  array of shape (n_features, n_classes)
              correlations between each feature and each target class
    """

    correlationVectors = []
    for classLabel in y:
        correlationVector = getCorrelationVector(x, classLabel)
        correlationVectors.append(correlationVector)

    return correlationVectors


def getCorrelationVector(x, y):
    """
    :param x: array of shape (n_samples, n_features)
              Training Data
    :param y: array of shape (n_samples)
              one-hot encoded array for target class
    :return:  array of shape (n_features)
              correlation between each feature and the target class
    """
    variableCorrelations = []
    # get transverse array
    npX = np.array(x).T

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

    num_correct_predictions = 0

    for i in range(0, len(targetVals)):
        if predictedVals[i] == targetVals[i]:
            num_correct_predictions = num_correct_predictions + 1

    accuracy = num_correct_predictions / len(targetVals)

    return accuracy


def deepClipVector(vector, c):
    """
    returns a copy of vector so that only absolute values greater than c are non-zero
    example: clippedVector([-0.9, 0.8, 0.1, 0, -0.4, 1], 0.5) returns [-0.9, 0.8, 0, 0, 0, 1]

    :param vector: array of shape (n_features)
    :param c: float value in the range (0,1)
              any value inside vector within the range [-c, c] will be set to 0.
    :return: array of shape (n_features)
             clipped vector
    """

    clippedVector = copy.deepcopy(vector)

    for i in range(0, len(clippedVector)):
        if abs(clippedVector[i]) <= c:
            clippedVector[i] = 0

    return clippedVector


def clipVectors(vectors, c):
    """
    Transforms inputted vector so that only absolute values greater than c are non-zero
    example: clippedVector([-0.9, 0.8, 0.1, 0, -0.4, 1], 0.5) returns [-0.9, 0.8, 0, 0, 0, 1]

    :param vectors: array of shape (num_classes, n_features)
                    correlation vectors
    :param c: float value in the range (0,1)
              any value inside vector within the range [-c, c] will be set to 0.
    """

    for vector in vectors:
        for i in range(0, len(vector)):
            if abs(vector[i]) <= c:
                vector[i] = 0


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
        self.num_classes = None
        # clipping value for correlation vector to remove noise
        self.clippingRange = 1
        # values predicted for unseen data
        self.predictions = None

    def fit(self, x, y, enable_clipping=False, max_iterations=10):
        """
        Fit the Correlation Discriminant Analysis model.

        :param x:              array-like of shape (n_samples, n_features)
                               Training data.
                               Assumes data in X is of normal distribution
                               requires categorical data to be one-hot encoded
                               requires no minority classes in X

        :param y:              array-like of shape (n_samples,)
                               Target values.

        :param enable_clipping: boolean
                               determines whether correlation vectors are transformed for optimisation

        :param max_iterations: integer
                               maximum gradient ascent iterations when finding optimal clipping values

        :return:               self : object
                               Fitted estimator.
        """

        encodedLabels = oneHotEncodedLabels(y)
        self.correlationVectors = getCorrelationVectors(x, encodedLabels)
        self.classes = unique_labels(y)
        self.num_classes = len(self.classes)

        if enable_clipping:
            clipping_range = self.__getOptimalClipping(x, y, max_iterations)
            clipVectors(self.correlationVectors, clipping_range)

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
        self.predictions = predictions

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
            max_strength = float('-inf')
            closest_class = -1
            for i in range(0, self.num_classes):
                classCorrelationVector = correlationVectors[i]
                class_strength = np.sum(np.dot(sample, classCorrelationVector))
                if class_strength > max_strength:
                    max_strength = class_strength
                    closest_class = i

            predictions.append(self.classes[closest_class])

        return predictions

    def __calcClippedAccuracy(self, x, c, y):
        """
        Calculates the accuracy of the correlation model when only using correlation values where the absolute value is
        above a given threshold c.

        :param x: training data
        :param c: clipping range
        :param y: target values for classes
        :return: accuracy using clipped vectors with range c
        """

        clippedVectors = []
        for vector in self.correlationVectors:
            clippedVector = deepClipVector(vector, c)
            clippedVectors.append(clippedVector)

        predictions = self.__correlationPredict(x, clippedVectors)

        accuracy = getAccuracy(predictions, y)

        return accuracy

    def __getOptimalClipping(self, x, y, max_iterations):
        """

        :param x:
        :param y:
        :param max_iterations:
        :return:
        """

        correlationFlattened = np.array(self.correlationVectors).flatten()
        correlationFlattened = [abs(ele) for ele in correlationFlattened]  # convert to absolute values
        correlationFlattened.sort()

        num_correlations = len(correlationFlattened)
        max_iterations = min(num_correlations, max_iterations)

        zero_clipping_accuracy = self.__calcClippedAccuracy(x, 0, y)  # get accuracy with no clipping

        best_index = 0
        best_accuracy = zero_clipping_accuracy

        checked = []  # array of indexes for which we have checked already
        num_spaced_checks = int(math.log2(num_correlations))

        for i in reversed(range(0, num_spaced_checks)):
            index = num_correlations - pow(2, i)
            clipping_point = correlationFlattened[index]
            checked.append(index)

            accuracy = self.__calcClippedAccuracy(x, clipping_point, y)
            best_accuracy = max(accuracy, best_accuracy)
            if accuracy == best_accuracy:
                best_index = index

            num_iterations = num_spaced_checks - i - 1

            if num_iterations >= max_iterations:
                return correlationFlattened[best_index]

        position = len(checked) - 1  # position when iterating through checked array

        for i in range(num_spaced_checks, max_iterations):
            if position == (len(checked) - 1):
                position = 0

                index = int(checked[0] / 2)
                clipping_point = correlationFlattened[index]
                checked.insert(0, index)

                accuracy = self.__calcClippedAccuracy(x, clipping_point, y)
                best_accuracy = max(accuracy, best_accuracy)
                if accuracy == best_accuracy:
                    best_index = index

                position = position + 1

                continue

            skip = False
            invalid_position = checked[position] - checked[position + 1] == -1

            while invalid_position:
                position = position + 1
                invalid_position = checked[position] - checked[position + 1] == -1
                if position == (len(checked) - 1):
                    i = i - 1
                    skip = True
                    invalid_position = False

            if skip:
                continue

            index = int((checked[position] + checked[position + 1]) / 2)
            checked.insert(position, index)
            position = position + 1

            clipping_point = correlationFlattened[index]

            accuracy = self.__calcClippedAccuracy(x, clipping_point, y)
            best_accuracy = max(accuracy, best_accuracy)
            if accuracy == best_accuracy:
                best_index = index

        return correlationFlattened[best_index]
