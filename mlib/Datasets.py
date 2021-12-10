import csv
import numpy as np

class Datasets:
    # IN : "./path/to/my/dataset.csv", "CollumnToPredict", [ "FeatureCollumn1", "FeatureCollumn2", ...]
    def __init__(self, filepath, predict_collumn, features_collumns):
        self.data = self.read_csv(filepath)
        self.predict = self.data[predict_collumn]
        self.size = len(self.predict)
        self.classes = sorted(set(self.data[predict_collumn]))
        self.nb_classes = len(self.classes)
        self.features = self.filter(features_collumns)
        self.nb_features = len(self.features)
        self.minmax = []
        self.scaled = None
        self.X = None
        self.Y = None
        self.X_train = None
        self.X_test = None
        self.Y_train = None
        self.Y_test = None
        self.train_size = None
        self.test_size = None

    # create a json from a csv filepath
    # IN : "./path/to/my/dataset.csv"
    # OUT: { 'a': [ "a1", "a2", ... ], 'b': [ "b1", "b2", ... ] , ... }
    def read_csv(self, filepath):
        data = None
        self.size = 0
        with open(filepath) as csvfile:
            reader = csv.reader(csvfile)
            for row in reader:
                if data is None:
                    labels = row
                    data = {label: [] for label in labels}
                else:
                    self.size += 1
                    for value, label in zip(row, labels):
                        data[label].append(value)
        return data

    # select only some json key
    # convert their values to float or NaN if not exist
    # remplace NaN by mean
    # IN : [ "a", "z" ]
    # OUT : {'a' : [ (float) a1, (float) a2 , ... ], 'z': [ (float) z1, (float) z2 , ... ]}
    def filter(self, keys):
        ret = []
        for label in self.data:
            if label in keys:
                ret.append([float(line) if line else np.NAN for line in self.data[label]])
        for i in range(len(keys)):
            for j in range(self.size):
                if np.isnan(ret[i][j]):
                    data = [ret[i][k] for k in range(self.size) if self.predict[j] == self.predict[k] and not np.isnan(ret[i][k])]
                    ret[i][j] = sum(data) / len(data)
        return ret

    # IN    :   [ 800,0, -800, ... ]
    # OUT   :   [ 1, 0.5, 0, ... ]
    def minmaxScaler(self, data, minmax=None):
        dmin = min(data) if minmax is None or min(data) < minmax['min'] else minmax['min']
        dmax = max(data) if minmax is None or max(data) > minmax['max'] else minmax['max']
        self.minmax.append({'min': dmin, 'max': dmax})
        return [((x - dmin) / (dmax - dmin)) for x in data]

    # IN    :   [ [ 800,0, -800, ... ] , ...]
    # OUT   :   [ [ 1, 0.5, 0, ... ] , ... ]
    def scale(self, minmax=None):
        self.scaled = []
        for i in range(self.nb_features):

            self.scaled.append(self.minmaxScaler(self.features[i], None if minmax is None else minmax[i]))
        self.X = self.transform(self.scaled)
        self.Y = self.get_Y(self.predict)
        return self.X, self.Y

    # IN    : [ [ a1, a2, a3, ... ], [ b1, b2, b3, ... ], [ c1, c2, c3, ... ], ... ]
    # OUT   : (np.array) [ [ a1, b1, c1, ... ], [ a2, b2, c2, ... ], [ a2, b2, c2, ... ], ... ]
    def transform(self, features):
        n = len(features)
        size = len(features[0])
        ret = np.ndarray((size, n), float)
        for i in range(n):
            for j in range(size):
                ret[j, i] = float(features[i][j])
        return ret

    # IN    : [ "class1", "class2", ... ]
    # OUT   : [ 1, 2, ... ]
    def get_Y(self, to_predict):
        Y = []
        for row in to_predict:
            Y.append(self.classes.index(row))
        return np.array(Y)

    # IN : [ [ a1, a2, a3, ... ], [ b1, b2, b3, ... ], [ c1, c2, c3, ... ], ... ]  , [0, 1, 2, ...]
    # OUT  [ [ a3, a1, a2, ... ], [ b3, b1, b2, ... ], [ c3, c1, c2, ... ], ... ]  , [2, 0, 1, ...]
    def shuffle(self, X, Y):
        rng_state = np.random.get_state()
        np.random.shuffle(X)
        np.random.set_state(rng_state)
        np.random.shuffle(Y)

    # Create X_test Y_test X_train and Y_train
    def train_test_split(self, percent_train):
        self.shuffle(self.X, self.Y)
        self.train_size = 100 - int(self.size * percent_train / 100)
        self.test_size = self.size - self.train_size
        self.Y_test = np.array(self.Y[:self.train_size])
        self.Y_train = np.array(self.Y[self.train_size:])
        self.X_test = np.array(self.X[:self.train_size])
        self.X_train = np.array(self.X[self.train_size:])



