import csv
from time import time

import numpy as np
from keras.callbacks import ModelCheckpoint
from keras.layers import Dense, Input, Concatenate, Lambda
from keras.models import Model, load_model, Sequential
from keras.utils import to_categorical
from tensorflow.python.keras.callbacks import TensorBoard

all_teams = ['BOS', 'LAC', 'CHI', 'PHI', 'CLE', 'MEM', 'ATL', 'SAC', 'UTA', 'POR', 'MIN', 'OKC', 'TOR', 'BKN', 'NOP',
             'GSW', 'MIA', 'DAL', 'PHX', 'IND', 'SAS', 'MIL', 'DET', 'NYK', 'CHA', 'WAS', 'HOU', 'LAL', 'DEN', 'ORL']


class AutoencoderNeuralNetwork:
    def __init__(self):
        self.model = None
        self.x_train = None
        self.x_test = None
        self.y_train = None
        self.y_test = None
        self.tensorboard = None
        self.team_dict = dict()
        for i in range(30):
            self.team_dict[all_teams[i]] = i
        #self.create_model()

    def get_data(self, path):
        x = list()
        y = list()
        with open(path) as file:
            reader = csv.reader(file, delimiter='|')
            for row in reader:
                x.append([float(row[i]) for i in range(143)])
                y.append(self.team_dict[row[-1]])
        normalize = np.array(x[:int(len(x)*0.9)])
        min_vec = np.min(normalize, axis=0)
        normalize = normalize - min_vec
        max_vec = np.max(normalize, axis=0)
        self.x_train = normalize / max_vec
        self.x_test = (np.array(x[int(len(x) * 0.9):])-min_vec)/max_vec
        ohv = to_categorical(np.array(y[:int(len(y) * 0.9)]), 30)
        self.y_train = ohv
        ohvt = to_categorical(np.array(y[int(len(y) * 0.9):]), 30)
        self.y_test = ohvt

    def create_model(self):
        wanted_dimensions = 10
        self.model = Sequential()
        self.model.add(Dense(8 * wanted_dimensions, input_dim=143, activation='relu'))
        self.model.add(Dense(4 * wanted_dimensions, activation='relu'))
        self.model.add(Dense(2*wanted_dimensions, activation='relu'))
        self.model.add(Dense(wanted_dimensions, activation='relu'))
        self.model.add(Dense(2 * wanted_dimensions, activation='relu'))
        self.model.add(Dense(4 * wanted_dimensions, activation='relu'))
        self.model.add(Dense(8 * wanted_dimensions, activation='relu'))
        self.model.add(Dense(143, activation='linear'))
        self.model.compile(optimizer='sgd',
                           loss="mean_squared_error")

    def fit_model(self):
        # checkpointer = ModelCheckpoint(filepath='../match_data/model_parameters_autoencoder.h5', verbose=1, save_best_only=True)
        #self.model.fit(self.x_train, self.x_train, epochs=35, batch_size=60, validation_data=(self.x_test, self.x_test),
        #               callbacks=[self.tensorboard, checkpointer])
        self.model.summary()
        self.tensorboard = TensorBoard(log_dir="logs/{}".format(time()))
        self.model.fit(self.x_train, self.y_train, epochs=90, batch_size=60, validation_data=(self.x_test, self.y_test),
                       callbacks=[self.tensorboard])

    def load_model(self):
        self.model = load_model('../match_data/model_parameters_autoencoder.h5')
        self.model.load_weights('../match_data/model_parameters_autoencoder.h5')
        self.model.pop()
        self.model.pop()
        self.model.pop()
        self.model.pop()
        for layer in self.model.layers:
            layer.trainable = False
        layer = Dense(30, activation='softmax')
        layer.name = layer.name + str("_new")
        self.model.add(layer)
        self.model.compile(loss='categorical_crossentropy', optimizer='sgd', metrics=['accuracy'])


class ClassificationNeuralNetwork:
    def __init__(self):
        self.team_dict = dict()
        for i in range(30):
            self.team_dict[all_teams[i]] = i
        self.model = None
        self.x_train = None
        self.y_train = None
        self.x_test = None
        self.y_test = None
        self.tensorboard = None
        self.create_model()

    def get_data(self, path):
        x = list()
        y = list()
        with open(path) as file:
            reader = csv.reader(file, delimiter='|')
            for row in reader:
                x.append([float(row[i]) for i in range(143)])
                y.append(self.team_dict[row[-1]])
        normalize = np.array(x[:int(len(x)*0.9)])
        min_vec = np.min(normalize, axis=0)
        normalize = normalize - min_vec
        max_vec = np.max(normalize, axis=0)
        self.x_train = normalize / max_vec
        ohv = to_categorical(np.array(y[:int(len(y)*0.9)]), 30)
        self.y_train = ohv
        self.x_test = (np.array(x[int(len(x) * 0.9):])-min_vec)/max_vec
        ohvt = to_categorical(np.array(y[int(len(y) * 0.9):]), 30)
        self.y_test = ohvt

    def create_model(self):
        # self.model = self.custom_classification_model()
        self.model = self.simple_classification_model()
        self.tensorboard = TensorBoard(log_dir="logs/{}".format(time()))
        self.model.compile(loss='categorical_crossentropy', optimizer='sgd', metrics=['accuracy'])

    def simple_classification_model(self):
        model = Sequential()
        model.add(Dense(72, input_dim=143))
        model.add(Dense(30, activation='softmax'))
        return model

    def custom_classification_model(self):
        input_tensor = Input((143,))
        # Moments
        moment_nodes = list()
        for i in range(10):
            nodes = [Dense(1)(Lambda(lambda x: x[:, (i * 14):(i * 14 + 4)], output_shape=(4,))(input_tensor))]
            for j in range(5):
                nodes.append(Dense(1)(
                    Lambda(lambda x: x[:, (i * 14 + j * 2 + 4):(i * 14 + j * 2 + 6)], output_shape=(2,))(input_tensor)))
            moment_nodes.append(Dense(1)(Concatenate()(nodes)))
        moment_nodes.append(Dense(1)(Lambda(lambda x: x[:, 140:143], output_shape=(3,))(input_tensor)))
        hidden = Dense(30)(Concatenate()(moment_nodes))
        hidden2 = Dense(5)(hidden)
        output_tensor = Dense(30, activation='softmax')(hidden2)
        return Model(input_tensor, output_tensor)

    def fit_model(self):
        checkpointer = ModelCheckpoint(filepath='../match_data/model_parameters_classification.h5', verbose=1, save_best_only=True)
        self.model.fit(self.x_train, self.y_train, epochs=40, batch_size=60, validation_data=(self.x_test, self.y_test),
                       callbacks=[checkpointer, self.tensorboard])
        self.model.summary()

    def load_model(self):
        self.model = load_model('../match_data/model_parameters_classification.h5')


class BallNeuralNetwork:
    def __init__(self):
        self.team_dict = dict()
        for i in range(30):
            self.team_dict[all_teams[i]] = i
        self.model = None
        self.x_train = None
        self.y_train = None
        self.x_test = None
        self.y_test = None
        self.tensorboard = None
        self.create_model()

    def get_data(self, path):
        x = list()
        y = list()
        with open(path) as file:
            reader = csv.reader(file, delimiter='|')
            for row in reader:
                x, y = self.transform_row(row, x, y)
        normalize = np.array(x[:int(len(x)*0.9)])
        min_vec = np.min(normalize, axis=0)
        normalize = normalize - min_vec
        max_vec = np.max(normalize, axis=0)
        normalize_y = np.array(y[:int(len(x) * 0.9)])
        min_vec_y = np.min(normalize_y, axis=0)
        normalize_y = normalize_y - min_vec_y
        max_vec_y = np.max(normalize_y, axis=0)
        self.x_train = normalize / max_vec
        self.y_train = normalize_y / max_vec_y
        self.x_test = (np.array(x[int(len(x) * 0.9):])-min_vec)/max_vec
        self.y_test = (np.array(y[int(len(y) * 0.9):])-min_vec_y)/max_vec_y

    def transform_row(self, row, x_list, y_list):
        i = 0
        x = list()
        y = list()
        while i < 10:
            y.append(float(row[i*14]))
            y.append(float(row[i * 14 + 1]))
            y.append(float(row[i * 14 + 2]))
            y.append(float(row[i * 14 + 3]))
            j = 0
            while j < 5:
                x.append(float(row[i * 14 + j * 2 + 4]))
                x.append(float(row[i * 14 + j * 2 + 5]))
                j += 1
            x_list.append(x)
            y_list.append(y)
            x = list()
            y = list()
            i += 1
        return x_list, y_list

    def create_model(self):
        self.model = self.simple_classification_model()
        self.tensorboard = TensorBoard(log_dir="logs/{}".format(time()))
        self.model.compile(optimizer='sgd',
                           loss="mean_squared_error")

    def simple_classification_model(self):
        model = Sequential()
        model.add(Dense(6, input_dim=10))
        model.add(Dense(4, activation='linear'))
        return model

    def fit_model(self):
        checkpointer = ModelCheckpoint(filepath='../match_data/model_parameters_ball.h5', verbose=1, save_best_only=True)
        self.model.fit(self.x_train, self.y_train, epochs=40, batch_size=60, validation_data=(self.x_test, self.y_test),
                       callbacks=[checkpointer, self.tensorboard])
        self.model.summary()

    def load_model(self):
        self.model = load_model('../match_data/model_parameters_ball.h5')


if __name__ == '__main__':
    # nn = NeuralNetwork()
    # nn.get_data('../match_data/nba_vectors_defenders_distance_corrected.csv')
    # # nn.load_model()
    # nn.fit_model()
    nn = AutoencoderNeuralNetwork()
    nn.get_data('../match_data/nba_vectors_defenders_distance_corrected.csv')
    nn.load_model()
    nn.fit_model()
