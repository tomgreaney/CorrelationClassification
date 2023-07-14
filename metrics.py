# Metrics Functions for CDA experiments
# Author:        Thomas Greaney <t9reaney@gmail.com>
# Created:       14th July 2023
# Last Modified: 14th July 2023

from sklearn.metrics import confusion_matrix
from sklearn.utils.multiclass import unique_labels
import numpy as np
import pandas as pd


class ConfusionMatrix:

    def __init__(self, true_labels, predicted_labels):
        self._class_labels = unique_labels(true_labels)
        confusion_sums = confusion_matrix(true_labels, predicted_labels)
        num_samples = len(true_labels)
        self._confusion_matrix = np.divide(confusion_sums, num_samples)

    def __str__(self):
        column_names = []
        row_names = []
        for label in self._class_labels:
            column_names.append("Predicted " + label)
            row_names.append("Truly " + label)
        df = pd.DataFrame(self._confusion_matrix, columns=column_names, index=row_names)

        title = "\n         Confusion Matrix\n"
        return title + str(df) + "\n"

    def printAccuracy(self):
        accuracy = 0
        for i in range(0, len(self._class_labels)):
            accuracy += self._confusion_matrix[i][i]

        print("Accuracy:", accuracy)
