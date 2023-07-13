# Correlation Discriminant Analysis test
# Author:        Thomas Greaney <t9reaney@gmail.com>
# Created:        7th July 2023
# Last Modified: 13th July 2023
from CDA import CorrelationDiscriminantAnalysis
import pandas as pd
from sklearn import preprocessing
import time

if __name__ == '__main__':
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

    random = df.sample(frac=1)
    train = random.head(6500)
    test = random.tail(1624)

    train_x = train.drop(["class"], axis=1).values.tolist()
    train_y = train["class"].to_numpy()

    cda = CorrelationDiscriminantAnalysis()
    start = time.time()
    cda.fit(train_x, train_y, enable_clipping=True, max_iterations=10)
    end = time.time()

    test_x = test.drop(["class"], axis=1).values.tolist()
    test_y = test["class"].to_numpy()

    predictions = cda.predict(test_x)

    positive = 'p'

    true_positives = 0
    true_negatives = 0
    false_positives = 0
    false_negatives = 0
    num_vals = len(predictions)

    for i in range(0, len(predictions)):
        if predictions[i] == positive:
            if predictions[i] == test_y[i]:
                true_positives = true_positives + 1
            else:
                false_positives = false_positives + 1
        else:
            if predictions[i] == test_y[i]:
                true_negatives = true_negatives + 1
            else:
                false_negatives = false_negatives + 1

    true_positives = round(true_positives / num_vals, 6)
    true_negatives = round(true_negatives / num_vals, 6)
    false_positives = round(false_positives / num_vals, 6)
    false_negatives = round(false_negatives / num_vals, 6)

    print("\nTime Taken:", str((end-start)*1000) + "ms.\n")

    print("                    Prediction Poisonous, Prediction Edible")
    print("Actually Poisonous ", true_positives, "            ", false_negatives)
    print("Actually Edible    ", false_positives, "            ", true_negatives)

    print("\nAccuracy:       ", round((true_positives + true_negatives), 6))
    print("Clipping Range: ", round(cda.clippingRange, 6))
