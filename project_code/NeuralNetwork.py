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
        self.x = None
        self.y = None
        self.tensorboard = None
        self.team_dict = dict()
        for i in range(30):
            self.team_dict[all_teams[i]] = i

    def get_data(self, path):
        self.x = list()
        self.y = list()
        with open(path) as file:
            reader = csv.reader(file, delimiter='|')
            for row in reader:
                self.x.append([float(row[i]) for i in range(143)])
                self.y.append(self.team_dict[row[-1]])
        normalize = np.array(self.x[:int(len(self.x)*0.8)])
        min_vec = np.min(normalize, axis=0)
        normalize = normalize - min_vec
        max_vec = np.max(normalize, axis=0)
        self.x_train = normalize / max_vec
        self.y_train = to_categorical(np.array(self.y[:int(len(self.y)*0.9)]), 30)
        self.x_test = (np.array(self.x[int(len(self.x) * 0.9):])-min_vec)/max_vec
        self.y_test = to_categorical(np.array(self.y[int(len(self.y) * 0.9):]), 30)

    def create_model(self):
        wanted_dimensions = 10
        self.model = Sequential()
        self.model.add(Dense(8 * wanted_dimensions, input_dim=143, activation='relu'))
        self.model.add(Dense(4 * wanted_dimensions, activation='relu'))
        self.model.add(Dense(2 * wanted_dimensions, activation='relu'))
        self.model.add(Dense(wanted_dimensions, activation='relu'))
        self.model.add(Dense(2 * wanted_dimensions, activation='relu'))
        self.model.add(Dense(4 * wanted_dimensions, activation='relu'))
        self.model.add(Dense(8 * wanted_dimensions, activation='relu'))
        self.model.add(Dense(143, activation='linear'))
        self.model.compile(optimizer='adam',
                           loss="mse")

    def fit_model(self):
        checkpointer = ModelCheckpoint(filepath='../match_data/model_parameters_vector_autoencoder.h5', verbose=1, save_best_only=True)
        self.tensorboard = TensorBoard(log_dir="logs/{}_vectorEncoder".format(time()))
        self.model.fit(self.x_train, self.x_train, epochs=200, batch_size=60, validation_data=(self.x_test, self.x_test),
                       callbacks=[self.tensorboard, checkpointer])

    def load_model(self):
        self.model = load_model('../match_data/model_parameters_vector_autoencoder.h5')
        self.model.load_weights('../match_data/model_parameters_vector_autoencoder.h5')
        vector_predictions = self.model.predict(self.x_test)
        print("MAE: " + str(np.sum(np.abs(vector_predictions - self.x_test))/vector_predictions.size))
        print(self.model.evaluate(x=self.x_test, y=self.x_test))
        print("")

    def classify_latent_vectors(self):
        # self.model.pop()
        # self.model.pop()
        # self.model.pop()
        # self.model.pop()
        # for layer in self.model.layers:
        #     layer.trainable = False
        # layer = Dense(30, activation='softmax')
        # layer.name = layer.name + str("_new")
        # self.model.add(layer)
        # self.model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        # checkpointer = ModelCheckpoint(filepath='../match_data/model_parameters_vector_classify.h5', verbose=1,
        #                                save_best_only=True)
        # self.tensorboard = TensorBoard(log_dir="logs/{}_vectorClassify".format(time()))
        # self.model.fit(self.x_train, self.y_train, epochs=400, batch_size=128,
        #                validation_data=(self.x_test, self.y_test),
        #                callbacks=[self.tensorboard, checkpointer])
        self.model = load_model('../match_data/model_parameters_vector_classify.h5')
        self.model.load_weights('../match_data/model_parameters_vector_classify.h5')
        pred = self.model.predict(self.x_test)
        self.test_brier(pred, self.y_test)
        self.test_accuracy(pred, self.y_test)
        print(self.model.evaluate(x=self.x_test, y=self.y_test))
        print("")

    def test_brier(self, predictions, real_value):
        preds = predictions - real_value
        s = np.sum(np.square(preds))
        print('brier: ' + str(s / predictions.shape[0]))

    def test_accuracy(self, predictions, real_value):
        predicted = np.max(np.multiply(real_value, predictions), axis=1)
        maxes = np.max(predictions, axis=1)
        s = np.sum(predicted == maxes)
        print('accuracy: ' + str(s / predictions.shape[0]))


class ClassificationNeuralNetwork:
    def __init__(self):
        self.team_dict = dict()
        for i in range(30):
            self.team_dict[all_teams[i]] = i
        self.model = None
        self.x = None
        self.y = None
        self.x_train = None
        self.y_train = None
        self.x_test = None
        self.y_test = None
        self.tensorboard = None

    def get_data(self, path):
        self.x = list()
        self.y = list()
        with open(path) as file:
            reader = csv.reader(file, delimiter='|')
            for row in reader:
                self.x.append([float(row[i]) for i in range(143)])
                self.y.append(self.team_dict[row[-1]])
        normalize = np.array(self.x[:int(len(self.x)*0.8)])
        min_vec = np.min(normalize, axis=0)
        normalize = normalize - min_vec
        max_vec = np.max(normalize, axis=0)
        self.x_train = normalize / max_vec
        self.y_train = to_categorical(np.array(self.y[:int(len(self.y)*0.8)]), 30)
        self.x_test = (np.array(self.x[int(len(self.x) * 0.8):])-min_vec)/max_vec
        self.y_test = to_categorical(np.array(self.y[int(len(self.y) * 0.8):]), 30)

    def create_model(self):
        self.model = self.custom_classification_model()
        # self.model = self.simple_classification_model()
        self.tensorboard = TensorBoard(log_dir="logs/{}_custom_vector_class".format(time()))
        self.model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    def simple_classification_model(self):
        model = Sequential()
        model.add(Dense(72, activation='relu', input_dim=143))
        model.add(Dense(30, activation='softmax'))
        return model

    def custom_classification_model(self):
        input_tensor = Input((143,))
        # Moments
        moment_nodes = list()
        for i in range(10):
            nodes = [Dense(1, activation='relu')(Lambda(lambda x: x[:, (i * 14):(i * 14 + 4)], output_shape=(4,))(input_tensor))]
            for j in range(5):
                nodes.append(Dense(1, activation='relu')(
                    Lambda(lambda x: x[:, (i * 14 + j * 2 + 4):(i * 14 + j * 2 + 6)], output_shape=(2,))(input_tensor)))
            moment_nodes.append(Dense(1, activation='relu')(Concatenate()(nodes)))
        moment_nodes.append(Dense(1, activation='relu')(Lambda(lambda x: x[:, 140:143], output_shape=(3,))(input_tensor)))
        hidden = Dense(30, activation='relu')(Concatenate()(moment_nodes))
        output_tensor = Dense(30, activation='softmax')(hidden)
        return Model(input_tensor, output_tensor)

    def fit_model(self):
        checkpointer = ModelCheckpoint(filepath='../match_data/custom_vector_model.h5', verbose=1, save_best_only=True)
        self.model.fit(self.x_train, self.y_train, epochs=200, batch_size=60,
                       validation_data=(self.x_test, self.y_test), callbacks=[checkpointer, self.tensorboard])
        pred = self.model.predict(self.x_test)
        self.test_brier(pred, self.y_test)
        self.test_accuracy(pred, self.y_test)
        print(self.model.evaluate(x=self.x_test, y=self.y_test))
        print("")

    def load_model(self):
        # self.model = load_model('../match_data/custom_vector_model.h5')
        # self.model.load_weights('../match_data/custom_vector_model.h5')
        self.model = load_model('../match_data/simple_vector_model.h5')
        self.model.load_weights('../match_data/simple_vector_model.h5')

    def test_vectors(self):
        pred = self.model.predict(self.x_test)
        self.test_brier(pred, self.y_test)
        self.test_accuracy(pred, self.y_test)
        print(self.model.evaluate(x=self.x_test, y=self.y_test))
        print("")

    def get_weights(self):
        self.model = load_model('../match_data/model_parameters_classification.h5')
        self.model.load_weights('../match_data/model_parameters_classification.h5')
        w = self.model.layers[0].get_weights()[0]
        with open("weights.csv", 'w') as file:
            h = [str(i) for i in range(72)]
            h.append('class')
            writer = csv.writer(file, delimiter=',')
            writer.writerow(h)
            for i in range(30):
                row = w[:, i].reshape(72).tolist()
                row.append(all_teams[i])
                writer.writerow(row)

    def test_brier(self, predictions, real_value):
        s = np.sum(np.square(predictions - real_value))
        print('brier: ' + str(s / predictions.shape[0]))

    def test_accuracy(self, predictions, real_value):
        predicted = np.max(np.multiply(real_value, predictions), axis=1)
        maxes = np.max(predictions, axis=1)
        s = np.sum(predicted == maxes)
        print('accuracy: ' + str(s / predictions.shape[0]))


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
    print("SIMPLE CLASIFICATION ------------------------------")
    nn = ClassificationNeuralNetwork()
    nn.get_data('../match_data/nba_vectors_defenders_distance_corrected.csv')
    nn.load_model()
    nn.test_vectors()
    # nn.get_weights()
    # nn.fit_model()
    print("VECTOR AUTOENCODER ------------------------------")
    nn = AutoencoderNeuralNetwork()
    nn.get_data('../match_data/nba_vectors_defenders_distance_corrected.csv')
    # nn.fit_model()
    nn.load_model()
    print("AUTOENCODER VECTOR CLASIFICATION ------------------------------")
    nn.classify_latent_vectors()

