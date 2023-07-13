# Correlation Discriminant Analysis
# Author:        Thomas Greaney <t9reaney@gmail.com>
# Created:        7th July 2023
# Last Modified: 13th July 2023
from sklearn.utils.multiclass import unique_labels
import math
import numpy as np
import utils


class CorrelationDiscriminantAnalysis:

    def __init__(self, n_components=None):
        self.n_components = n_components  # number of features to use when training
        self.correlationVectors = None  # correlation vectors to use for unseen data
        self.classes = None  # class labels
        self.num_classes = None  # number of classes in prediction
        self.clippingRange = 1  # clipping value for correlation vector to remove noise
        self.predictions = None  # values predicted for unseen data

    def fit(self, x, y, enable_clipping=False, max_iterations=10, variance_scaling=False):
        """
        Fit the Correlation Discriminant Analysis model.

        :param x:               array-like of shape (n_samples, n_features)
                                Training data.
                                Assumes data in X is of normal distribution
                                requires categorical data to be one-hot encoded
                                requires no minority classes in X

        :param y:               array-like of shape (n_samples,)
                                Target values.

        :param enable_clipping: boolean
                                determines whether correlation vectors are transformed for optimisation

        :param max_iterations:   integer
                                 maximum gradient ascent iterations when finding optimal clipping values

        :param variance_scaling: boolean
                                 applies scaling to correlation vectors based off the variance between classes

        :return:                self : object
                                Fitted estimator.
        """

        y = np.array(y)
        x = np.array(x)

        encodedLabels = utils.oneHotEncodedLabels(y)
        self.correlationVectors = utils.getCorrelationVectors(x, encodedLabels)
        self.classes = unique_labels(y)
        self.num_classes = len(self.classes)

        if variance_scaling:
            variances = utils.getVariances(self.correlationVectors)
            for i in range(0, self.num_classes):
                self.correlationVectors[i] = np.multiply(variances, self.correlationVectors[i])

        if enable_clipping:
            clipping_range = self.__getOptimalClipping(x, y, max_iterations)
            self.clippingRange = clipping_range
            utils.clipVectors(self.correlationVectors, clipping_range)

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

        self.predictions = predictions

        return np.array(predictions)

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
            clippedVector = utils.deepClipVector(vector, c)
            clippedVectors.append(clippedVector)

        predictions = self.__correlationPredict(x, clippedVectors)

        accuracy = utils.getAccuracy(predictions, y)

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

        # optimise this by just checking indices in proximity to best index
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
                if position == (len(checked) - 1):
                    i = i - 1
                    skip = True
                    invalid_position = False
                if invalid_position:
                    invalid_position = checked[position] - checked[position + 1] == -1

            if skip:
                continue

            index = int((checked[position] + checked[position + 1]) / 2)
            checked.insert(position + 1, index)
            position = position + 2

            clipping_point = correlationFlattened[index]

            accuracy = self.__calcClippedAccuracy(x, clipping_point, y)
            best_accuracy = max(accuracy, best_accuracy)
            if accuracy == best_accuracy:
                best_index = index

        return correlationFlattened[best_index]
