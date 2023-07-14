# Correlation Discriminant Analysis test
# Author:        Thomas Greaney <t9reaney@gmail.com>
# Created:        7th July 2023
# Last Modified: 13th July 2023
from CDA import CorrelationDiscriminantAnalysis
import pandas as pd
from sklearn import preprocessing
import time
from metrics import ConfusionMatrix

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
    cda.fit(train_x, train_y, enable_clipping=True, max_iterations=30, variance_scaling=True)
    end = time.time()

    test_x = test.drop(["class"], axis=1).values.tolist()
    test_y = test["class"].to_numpy()

    predictions = cda.predict(test_x)

    print("Clipping Range: ", round(cda.clippingRange, 6))

    confusionMatrix = ConfusionMatrix(test_y, predictions)
    print(confusionMatrix)

    confusionMatrix.printAccuracy()
