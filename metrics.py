# Metrics Functions for CDA experiments
# Author:        Thomas Greaney <t9reaney@gmail.com>
# Created:       14th July 2023
# Last Modified: 20th July 2023

from sklearn.metrics import confusion_matrix
from sklearn.utils.multiclass import unique_labels
import numpy as np
import pandas as pd
import copy


class ConfusionMatrix:

    def __init__(self, true_labels, predicted_labels, title):
        confusion_sums = confusion_matrix(predicted_labels, true_labels)
        num_samples = len(true_labels)
        self._class_labels = unique_labels(true_labels)
        self._confusion_matrix = np.divide(confusion_sums, num_samples)
        self._title = title

    def __str__(self):
        column_names = []
        row_names = []
        for label in self._class_labels:
            column_names.append("Predicted " + str(label))
            row_names.append("Truly " + str(label))
        df = pd.DataFrame(self._confusion_matrix, columns=column_names, index=row_names)

        title = "\n         " + self._title + "\n"
        return title + df.to_string(max_cols=10) + "\n"

    def printAccuracy(self):
        accuracy = 0
        for i in range(0, len(self._class_labels)):
            accuracy += self._confusion_matrix[i][i]

        print("Accuracy:", accuracy)

    def getAccuracy(self):
        accuracy = 0
        for i in range(0, len(self._class_labels)):
            accuracy += self._confusion_matrix[i][i]

        return accuracy

    def getMatrix(self):
        return copy.deepcopy(self._confusion_matrix)


def getZScore(x1, x2, label):
    if len(x1) != len(x2):
        raise Exception("Comparing two arrays of unequal size")

    differences = [0] * len(x1)
    for i in range(0, len(differences)):
        differences[i] = x1[i] - x2[i]

    mean = np.mean(differences)
    std = np.std(differences)

    z_score = mean / std

    return z_score
