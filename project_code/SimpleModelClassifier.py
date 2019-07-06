import csv

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import ShuffleSplit, cross_val_score


class SimpleModel:
    def __init__(self):
        self.x = list()
        self.y = list()
        self.okc = 0
        self.gsw = 0
        self.mem = 0
        self.clf = RandomForestClassifier(100)
        self.read_file()

    def read_file(self):
        with open('../match_data/nba_vectors_defenders_distance_corrected.csv') as csvfile:
            reader = csv.reader(csvfile, delimiter='|')
            for row in reader:
                if len(row) < 147 or float(row[144]) > 240:  # points only 44, defenders distance 144
                    continue
                x = [float(row[i]) for i in range(143)]  # points only 43, defenders distance 143
                self.add_row(x, row[-1])

    def add_row(self, x, y):
        self.x.append(x)
        self.y.append(y)

    def create_a_model(self):
        x = np.array(self.x)
        y = np.array(self.y)
        cv = ShuffleSplit(n_splits=10, random_state=None, test_size=0.1, train_size=None)
        cv_score = cross_val_score(self.clf, x, y, cv=cv)
        print(np.mean(cv_score))


if __name__ == '__main__':
    m = SimpleModel()
    m.create_a_model()
