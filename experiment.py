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


def run_adults_experiment(num_iterations=10):
    print("\nTesting Correlation Discriminant Analysis on Diabetes Dataset")
    print("6 models are evaluated using the same experiment repeated", num_iterations, "times.")
    print("Each repetition has an independent random split on training/test data, with an 80:20 split ratio.\n")

    # prepare data
    df = pd.read_csv('data/adult.csv')

    # make even distribution between over and under 50k diagnosis
    over = df[df['salaryOver50K'] == 1]
    under = df[df['salaryOver50K'] == 0].sample(n=len(over.index))
    df = pd.concat([over, under])

    df = pd.get_dummies(df,
                        columns=['workclass', 'education', 'marital-status', 'occupation', 'relationship', 'race',
                                 'sex', 'capital-gain', 'capital-loss', 'native-country'],
                        drop_first=False)

    ageStd = df['age'].std()
    ageMean = df['age'].mean()
    cutOff1 = ageMean - (2 * ageStd)

    def condition1(x):
        if x < cutOff1:
            return 1
        else:
            return 0

    cutOff2 = ageMean - (0.5 * ageStd)

    def condition2(x):
        if x < cutOff2 and x >= cutOff1:
            return 1
        else:
            return 0

    cutOff3 = ageMean + (0.5 * ageStd)

    def condition3(x):
        if x < cutOff3 and x >= cutOff2:
            return 1
        else:
            return 0

    cutOff4 = ageMean + (2 * ageStd)

    def condition4(x):
        if x < cutOff4 and x >= cutOff3:
            return 1
        else:
            return 0

    def condition5(x):
        if x >= cutOff4:
            return 1
        else:
            return 0

    # df['age1'] = df['age'].apply(condition1) 0 exist
    df['age2'] = df['age'].apply(condition2)
    df['age3'] = df['age'].apply(condition3)
    df['age4'] = df['age'].apply(condition4)
    df['age5'] = df['age'].apply(condition5)
    df = df.drop(columns=['age'])

    fnlwgtStd = df['fnlwgt'].std()
    fnlwgtMean = df['fnlwgt'].mean()
    cutOff1 = fnlwgtMean - (2 * fnlwgtStd)
    cutOff2 = fnlwgtMean - (0.5 * fnlwgtStd)
    cutOff3 = fnlwgtMean + (0.5 * fnlwgtStd)
    cutOff4 = fnlwgtMean + 2 * fnlwgtStd
    # df['fnlwgt1'] = df['fnlwgt'].apply(condition1) none exist
    df['fnlwgt2'] = df['fnlwgt'].apply(condition2)
    df['fnlwgt3'] = df['fnlwgt'].apply(condition3)
    df['fnlwgt4'] = df['fnlwgt'].apply(condition4)
    df['fnlwgt5'] = df['fnlwgt'].apply(condition5)
    df = df.drop(columns=['fnlwgt'])

    educationStd = df['education-num'].std()
    educationMean = df['education-num'].mean()
    cutOff1 = educationMean - (2 * educationStd)
    cutOff2 = educationMean - (0.5 * educationStd)
    cutOff3 = educationMean + (0.5 * educationStd)
    cutOff4 = educationMean + 2 * educationStd
    # df['education-num1'] = df['education-num'].apply(condition1)
    df['education-num2'] = df['education-num'].apply(condition2)
    df['education-num3'] = df['education-num'].apply(condition3)
    df['education-num4'] = df['education-num'].apply(condition4)
    # df['education-num5'] = df['education-num'].apply(condition5)
    df = df.drop(columns=['education-num'])

    hoursStd = df['hours-per-week'].std()
    hoursMean = df['hours-per-week'].mean()
    cutOff1 = hoursMean - (2 * hoursStd)
    cutOff2 = hoursMean - (0.5 * hoursStd)
    cutOff3 = hoursMean + (0.5 * hoursStd)
    cutOff4 = hoursMean + 2 * hoursStd
    df['hours-per-week1'] = df['hours-per-week'].apply(condition1)
    df['hours-per-week2'] = df['hours-per-week'].apply(condition2)
    df['hours-per-week3'] = df['hours-per-week'].apply(condition3)
    df['hours-per-week4'] = df['hours-per-week'].apply(condition4)
    df['hours-per-week5'] = df['hours-per-week'].apply(condition5)
    df = df.drop(columns=['hours-per-week'])

    # make even distribution between healthy and diabetes diagnosis
    experiment_results = []
    test_ys = []
    accuracies = [[], [], [], [], [], []]

    # get average accuracies and confusion matrices over num_iterations experiments
    for i in range(0, num_iterations):
        print("Running Experiments:", str(i) + "/" + str(num_iterations))
        random = df.sample(frac=1)
        train = random.head(26000)
        test = random.tail(6000)

        train_x = train.drop(["salaryOver50K"], axis=1).values.tolist()
        train_y = train["salaryOver50K"].to_numpy()

        test_x = test.drop(["salaryOver50K"], axis=1).values.tolist()
        test_y = test["salaryOver50K"].to_numpy()

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
            elif j > i:
                z_score = getZScore(accuracies[j], accuracies[i], str(j + 1) + " " + str(i + 1))
                p_value = stats.t.sf(abs(z_score), df=(num_iterations - 1)) * 2
                confidence = 100 * (1 - p_value)
                confidence_better[i][j] = str(round(confidence, 2))

    confidence_matrix = pd.DataFrame(confidence_better, columns=columns, index=columns)
    print("\nConfidence percentage that one model outperforms another")
    print(confidence_matrix)


def run_cover_experiment(num_iterations=10):
    print("\nTesting Correlation Discriminant Analysis on Mushroom Dataset")
    print("6 models are evaluated using the same experiment repeated", num_iterations, "times.")
    print("Each repetition has an independent random split on training/test data, with an 80:20 split ratio.\n")

    # prepare data
    df = pd.read_csv('data/covtype.csv', dtype={'Cover_Type': str})

    constant = (2 * np.pi) / 360

    def cos(x):
        return np.cos(x * constant)

    def sin(x):
        return np.sin(x * constant)

    constant = (2 * np.pi) / 360
    df['sin(Aspect)'] = df['Aspect'].apply(sin)
    df['cos(Aspect)'] = df['Aspect'].apply(cos)
    df = df.drop(columns=['Aspect'])

    min_max_scaler = preprocessing.MinMaxScaler()
    df[['Elevation']] = min_max_scaler.fit_transform(df[['Elevation']])
    df[['Slope']] = min_max_scaler.fit_transform(df[['Slope']])
    df[['Hydro_Vert_Dist']] = min_max_scaler.fit_transform(df[['Hydro_Vert_Dist']])
    df[['Hydro_Hor_Dist']] = min_max_scaler.fit_transform(df[['Hydro_Hor_Dist']])
    df[['Road_Hor_Dist']] = min_max_scaler.fit_transform(df[['Road_Hor_Dist']])
    df[['Shade_09']] = min_max_scaler.fit_transform(df[['Shade_09']])
    df[['Shade_12']] = min_max_scaler.fit_transform(df[['Shade_12']])
    df[['Shade_15']] = min_max_scaler.fit_transform(df[['Shade_15']])
    df[['FireP_Hor_Dist']] = min_max_scaler.fit_transform(df[['FireP_Hor_Dist']])
    df[['sin(Aspect)']] = min_max_scaler.fit_transform(df[['sin(Aspect)']])
    df[['cos(Aspect)']] = min_max_scaler.fit_transform(df[['cos(Aspect)']])

    cover4 = df[df["Cover_Type"] == "4"]
    sampleLength = len(cover4.index)

    cover1 = df[df['Cover_Type'] == "1"].sample(n=sampleLength)
    cover2 = df[df['Cover_Type'] == "2"].sample(n=sampleLength)
    cover3 = df[df['Cover_Type'] == "3"].sample(n=sampleLength)
    cover5 = df[df['Cover_Type'] == "5"].sample(n=sampleLength)
    cover6 = df[df['Cover_Type'] == "6"].sample(n=sampleLength)
    cover7 = df[df['Cover_Type'] == "7"].sample(n=sampleLength)

    df = pd.concat([cover1, cover2, cover3, cover4, cover5, cover6, cover7])

    dropped = []

    for column in df:
        if df[column].nunique() == 1:
            df = df.drop(columns=[column])
            dropped.append(column)

    print("dropped:", dropped)

    experiment_results = []
    test_ys = []
    accuracies = [[], [], [], [], [], []]

    i = 0

    # get average accuracies and confusion matrices over num_iterations experiments
    while i < num_iterations:
        i = i + 1
        random = df.sample(frac=1)
        train = random.head(15383)
        skip = False
        for column in train:
            if train[column].nunique() == 1:
                train = train.drop(columns=[column])
                dropped.append(column)
                skip = True
        if skip:
            i = i - 1
            continue

        test = random.tail(3846)
        print("Running Experiments:", str(i) + "/" + str(num_iterations))


        train_x = train.drop(['Cover_Type'], axis=1).values.tolist()
        train_y = train['Cover_Type'].to_numpy()

        test_x = test.drop(['Cover_Type'], axis=1).values.tolist()
        test_y = test['Cover_Type'].to_numpy()

        test_ys.append(test_y)
        experiment_result = run_experiment(train_x, train_y, test_x)
        experiment_accuracies = [ConfusionMatrix(result, test_y, "Confusion Matrix").getAccuracy() for result in
                                 experiment_result]
        for j in range(0, len(experiment_accuracies)):
            accuracies[j].append(experiment_accuracies[j])

        experiment_results.append(experiment_result)


    printExperimentResults(test_ys, experiment_results, accuracies, num_iterations)
