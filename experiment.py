# Classification Algorithm Experiment Skeleton
# Author:        Thomas Greaney <t9reaney@gmail.com>
# Created:       19th July 2023
# Last Modified: 20th July 2023

from CDA import CorrelationDiscriminantAnalysis
from metrics import ConfusionMatrix, getZScore

from sklearn import preprocessing
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.neighbors import KNeighborsClassifier

import pandas as pd
import numpy as np
from scipy import stats


def run_experiment(train_x, train_y, test_x):
    """
    harness for experiments

    :param train_x:
    :param train_y:
    :param test_x:
    :return:
    """
    cda = CorrelationDiscriminantAnalysis()

    # CDA Predict without variance scaling, without clipping
    cda.fit(train_x, train_y, variance_scaling=False, enable_clipping=False)
    cda_raw_results = cda.predict(test_x)

    # CDA Predict with variance scaling, without clipping
    cda.fit(train_x, train_y, variance_scaling=True, enable_clipping=False)
    cda_variance_results = cda.predict(test_x)

    # CDA Predict with variance scaling, with clipping
    cda.fit(train_x, train_y, variance_scaling=True, enable_clipping=True)
    cda_clipping_results = cda.predict(test_x)

    # CDA Predict with variance scaling, with clipping, with 20 iterations
    cda.fit(train_x, train_y, variance_scaling=True, enable_clipping=True, max_iterations=20)
    cda_iterations_results = cda.predict(test_x)

    # LDA predict
    lda = LinearDiscriminantAnalysis()
    lda.fit(train_x, train_y)
    lda_predictions = lda.predict(test_x)

    # K Neighbours predict
    knn = KNeighborsClassifier()
    knn.fit(train_x, train_y)
    knn_predictions = knn.predict(test_x)

    results = [cda_raw_results, cda_variance_results, cda_clipping_results, cda_iterations_results, lda_predictions,
               knn_predictions]

    return results


def run_mushroom_experiment(num_iterations=10):

    print("\nTesting Correlation Discriminant Analysis on Mushroom Dataset")
    print("6 models are evaluated using the same experiment repeated", num_iterations, "times.")
    print("Each repetition has an independent random split on training/test data, with an 80:20 split ratio.\n")

    # prepare data
    df = pd.read_csv('data/mushrooms.csv', dtype={'class': str})
    df = df.drop(columns=['veil-type', 'stalk-root'])

    ring_no_scale_mapper = {
        "n": 0,
        "o": 1,
        "t": 2
    }
    df['ring-number'] = df['ring-number'].replace(ring_no_scale_mapper)

    # One hot encode columns

    df = pd.get_dummies(df,
                        columns=['cap-shape', 'cap-surface', 'cap-color', 'bruises',
                                 'odor', 'gill-attachment', 'gill-size', 'gill-color',
                                 'stalk-shape', 'stalk-surface-above-ring', 'stalk-surface-below-ring',
                                 'stalk-color-above-ring', 'stalk-color-below-ring',
                                 'veil-color', 'ring-type', 'spore-print-color', 'habitat',
                                 'population', 'gill-spacing'],
                        drop_first=False)

    # rescale data
    min_max_scaler = preprocessing.MinMaxScaler()
    df[['ring-number']] = min_max_scaler.fit_transform(df[['ring-number']])

    experiment_results = []
    test_ys = []
    accuracies = [[], [], [], [], [], []]

    # get average accuracies and confusion matrices over num_iterations experiments
    for i in range(0, num_iterations):
        print("Running Experiments:", str(i) + "/" + str(num_iterations))
        random = df.sample(frac=1)
        train = random.head(6500)
        test = random.tail(1624)

        train_x = train.drop(["class"], axis=1).values.tolist()
        train_y = train["class"].to_numpy()

        test_x = test.drop(["class"], axis=1).values.tolist()
        test_y = test["class"].to_numpy()

        test_ys.append(test_y)
        experiment_result = run_experiment(train_x, train_y, test_x)
        experiment_accuracies = [ConfusionMatrix(result, test_y, "Confusion Matrix").getAccuracy() for result in
                                 experiment_result]
        for j in range(0, len(experiment_accuracies)):
            accuracies[j].append(experiment_accuracies[j])

        experiment_results.append(experiment_result)

    printExperimentResults(test_ys, experiment_results, accuracies, num_iterations)


def printExperimentResults(test_ys, experiment_results, accuracies, num_iterations):
    test_y = np.array(test_ys).flatten()

    cda_raw_results = [result[0] for result in experiment_results]
    cda_raw_results = np.array(cda_raw_results).flatten()
    cda_raw_matrix = ConfusionMatrix(cda_raw_results, test_y, "Model 1 Average Confusion Matrix")

    cda_variance_results = [result[1] for result in experiment_results]
    cda_variance_results = np.array(cda_variance_results).flatten()
    cda_variance_matrix = ConfusionMatrix(cda_variance_results, test_y, "Model 2 Average Confusion Matrix")

    cda_clipping_results = [result[2] for result in experiment_results]
    cda_clipping_results = np.array(cda_clipping_results).flatten()
    cda_clipping_matrix = ConfusionMatrix(cda_clipping_results, test_y, "Model 3 Average Confusion Matrix")

    cda_iterations_results = [result[3] for result in experiment_results]
    cda_iterations_results = np.array(cda_iterations_results).flatten()
    cda_iterations_matrix = ConfusionMatrix(cda_iterations_results, test_y, "Model 4 Average Confusion Matrix")

    lda_results = [result[4] for result in experiment_results]
    lda_results = np.array(lda_results).flatten()
    lda_matrix = ConfusionMatrix(lda_results, test_y, "Model 5 Average Confusion Matrix")

    knn_results = [result[5] for result in experiment_results]
    knn_results = np.array(knn_results).flatten()
    knn_matrix = ConfusionMatrix(knn_results, test_y, "Model 6 Average Confusion Matrix")

    print(cda_raw_matrix)
    print(cda_variance_matrix)
    print(cda_clipping_matrix)
    print(cda_iterations_matrix)
    print(lda_matrix)
    print(knn_matrix)

    min_accuracies = [min(accuracies[0]), min(accuracies[1]), min(accuracies[2]), min(accuracies[3]),
                      min(accuracies[4]), min(accuracies[5])]
    max_accuracies = [max(accuracies[0]), max(accuracies[1]), max(accuracies[2]), max(accuracies[3]),
                      max(accuracies[4]), max(accuracies[5])]
    avg_accuracies = [cda_raw_matrix.getAccuracy(), cda_variance_matrix.getAccuracy(),
                      cda_clipping_matrix.getAccuracy(), cda_iterations_matrix.getAccuracy(), lda_matrix.getAccuracy(),
                      knn_matrix.getAccuracy()]

    values = [avg_accuracies, min_accuracies, max_accuracies]

    columns = ["Model 1", "Model 2", "Model 3", "Model 4", "Model 5", "Model 6"]
    titles = ["Average Accuracy", "Minimum Accuracy", "Maximum Accuracy"]

    df = pd.DataFrame(values, columns=columns, index=titles)

    print("Accuracy information for each model across", num_iterations, "experiments\n")
    print(df)

    NUM_MODELS = 6

    confidence_better = [[""] * NUM_MODELS for i in range(NUM_MODELS)]

    for i in range(0, NUM_MODELS):
        for j in range(0, NUM_MODELS):
            if j == i:
                confidence_better[i][j] = "X"
            elif j < i:
                confidence_better[i][j] = ""
            else:
                z_score = getZScore(accuracies[j], accuracies[i], str(j+1) + " " + str(i+1))
                p_value = stats.t.sf(abs(z_score), df=(num_iterations-1)) * 2
                confidence = 100 * (1 - p_value)
                confidence_better[i][j] = str(round(confidence, 2))

    confidence_matrix = pd.DataFrame(confidence_better, columns=columns, index=columns)
    print("\nConfidence percentage that one model outperforms another")
    print(confidence_matrix)

