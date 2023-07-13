# Correlation Discriminant Analysis
# Author:        Thomas Greaney <t9reaney@gmail.com>
# Created:       13th July 2023
# Last Modified: 13th July 2023

from sklearn.utils.multiclass import unique_labels
import copy
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

    for feature in npX:
        # calculate pearson correlation
        correlation = stats.pearsonr(feature, y)[0]
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
    """
    Get the percentage of correct predictions

    :param predictedVals: array of shape (n_samples)
                          class labels of predicted classes
    :param targetVals:    array of shape (n_samples)
                          class labels of true classes
    :return:              float
                          accuracy value
    """

    if len(predictedVals) != len(targetVals):
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

