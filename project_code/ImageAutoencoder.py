import csv
import glob
from time import time

import numpy as np
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
from keras.layers import Conv2D, MaxPooling2D, UpSampling2D, Flatten, Reshape, Input, Dense
from keras.models import Sequential, load_model, Model
from keras.preprocessing.image import img_to_array, load_img
from keras.utils import to_categorical
from matplotlib import pyplot as plt
from tensorflow.python.keras.callbacks import TensorBoard


class ImageAutoencoder:
    def __init__(self):
        self.model = None
        self.x_train = None
        self.x_test = None
        self.y = None
        self.x = None
        self.tensorboard = None
        self.get_data()

    def get_data(self):
        data = list()
        self.y = list()
        for file in glob.glob('../match_data/images/*'):
            image = img_to_array(load_img(file))[40:, :, 0]
            tmp = np.zeros((54, 54))
            tmp[:, 2:-2] = image
            data.append(tmp)
            self.y .append((file.split('/')[-1]).split('_')[0])
        #random.shuffle(data)
        self.x = np.array(data)/255
        self.x = np.expand_dims(self.x, axis=-1)
        self.x_train = np.array(data[:int(len(data) * 0.9)])
        self.x_test = np.array(data[int(len(data) * 0.9):])
        self.x_train[self.x_train < 55] = 0
        self.x_test[self.x_test < 55] = 0
        self.x_train /= 255
        self.x_test /= 255
        self.x_train = np.expand_dims(self.x_train, axis=-1)
        self.x_test = np.expand_dims(self.x_test, axis=-1)

    def create_model(self):
        self.model = Sequential()
        self.model.add(Conv2D(32, (3, 3), padding='same', activation='relu', input_shape=(54, 54, 1)))
        self.model.add(MaxPooling2D((3, 3), padding='same'))
        #self.model.add(Conv2D(32, (3, 3), activation='relu', padding='same'))
        #self.model.add(MaxPooling2D((2, 2), padding='same'))
        self.model.add(Conv2D(16, (3, 3), activation='relu', padding='same'))
        self.model.add(MaxPooling2D((3, 3), padding='same'))
        #self.model.add(Conv2D(8, (3, 3), activation='relu', padding='same'))
        #self.model.add(MaxPooling2D((2, 2), padding='same'))
        self.model.add(Conv2D(8, (3, 3), strides=(2, 2), activation='relu', padding='same'))
        self.model.add(Flatten())
        self.model.add(Reshape((3, 3, 8)))
        self.model.add(UpSampling2D((2, 2)))
        self.model.add(Conv2D(8, (3, 3), activation='relu', padding='same'))
        #self.model.add(UpSampling2D((2, 2)))
        #self.model.add(Conv2D(8, (3, 3), activation='relu', padding='same'))
        self.model.add(UpSampling2D((3, 3)))
        self.model.add(Conv2D(16, (3, 3), activation='relu', padding='same'))
        #self.model.add(UpSampling2D((2, 2)))
        #self.model.add(Conv2D(32, (3, 3), activation='relu', padding='same'))
        self.model.add(UpSampling2D((3, 3)))
        self.model.add(Conv2D(32, (3, 3), activation='relu', padding='same'))
        self.model.add(Conv2D(1, (3, 3), activation='linear', padding='same'))
        self.model.compile(optimizer='adam', loss='mse')
        self.model.summary()

    def fit_model(self):
        checkpointer = ModelCheckpoint(filepath='../match_data/model_parameters_image_autoencoder3.h5', verbose=1,
                                       save_best_only=True)
        reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5,
                                      patience=3, min_lr=0.001)
        tensorboard = TensorBoard(log_dir="logs/{}".format(time()))
        self.model.fit(self.x_train, self.x_train, epochs=20, batch_size=128,
                       validation_data=(self.x_test, self.x_test), callbacks=[checkpointer, reduce_lr, tensorboard])

    def load_model(self):
        self.model = load_model('../match_data/model_parameters_image_autoencoder2.h5')
        self.model.load_weights('../match_data/model_parameters_image_autoencoder2.h5')

    def predict(self):
        random_test_images = np.random.randint(self.x_test.shape[0], size=10)

        decoded_imgs = self.model.predict(self.x_test)

        plt.figure(figsize=(18, 4))

        for i, image_idx in enumerate(random_test_images):
            # plot original image
            ax = plt.subplot(3, 10, i + 1)
            plt.imshow(self.x_test[image_idx].reshape(54, 54))
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)

            # plot reconstructed image
            ax = plt.subplot(3, 10, 2 * 10 + i + 1)
            plt.imshow(decoded_imgs[image_idx].reshape(54, 54))
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)
        plt.show()

    def show_kernels(self):
        layer = Input(self.model.layers[-8].input_shape[1:])
        n_l = self.model.layers[-8](layer)
        for i in reversed(range(1, 8)):
            n_l = self.model.layers[-i](n_l)
        new_model = Model(layer, n_l)
        new_model.summary()
        pred = np.array([
            [4.1,4.1,4.1,4.1,4.1,4.1,4.1,4.1,4.1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
            [0,0,0,0,0,0,0,0,0,4.1,4.1,4.1,4.1,4.1,4.1,4.1,4.1,4.1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
            [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,4.1,4.1,4.1,4.1,4.1,4.1,4.1,4.1,4.1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
            [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,4.1,4.1,4.1,4.1,4.1,4.1,4.1,4.1,4.1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
            [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,4.1,4.1,4.1,4.1,4.1,4.1,4.1,4.1,4.1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
            [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,4.1,4.1,4.1,4.1,4.1,4.1,4.1,4.1,4.1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
            [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,4.1,4.1,4.1,4.1,4.1,4.1,4.1,4.1,4.1,0,0,0,0,0,0,0,0,0],
            [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,4.1,4.1,4.1,4.1,4.1,4.1,4.1,4.1,4.1]
        ])
        decoded = new_model.predict(pred)
        plt.figure(figsize=(18, 4))
        for i in range(8):
            # plot original image
            ax = plt.subplot(3, 8, i + 1)
            plt.imshow(decoded[i].reshape(54, 54)*255)
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)
        plt.show()

    def make_data_from_network(self):
        self.model.pop()
        self.model.pop()
        self.model.pop()
        self.model.pop()
        self.model.pop()
        self.model.pop()
        self.model.pop()
        self.model.pop()
        for layer in self.model.layers:
            layer.trainable = False
        self.model.compile(optimizer='adam', loss='mse')
        self.model.summary()
        predictions = self.model.predict(self.x)

        with open("image_autoencoder_vectors.csv", 'w') as file:
            writer = csv.writer(file, delimiter=',')
            for i in range(len(predictions)):
                row = predictions[i, :].tolist()
                row.append(self.y[i])
                writer.writerow(row)

    def show_attack(self, filename):
        layer = Input(self.model.layers[-8].input_shape[1:])
        n_l = self.model.layers[-8](layer)
        for i in reversed(range(1, 8)):
            n_l = self.model.layers[-i](n_l)
        new_model = Model(layer, n_l)
        new_model.summary()
        data = list()
        with open(filename) as file:
            reader = csv.reader(file, delimiter=',')
            for row in reader:
                data.append([float(row[i]) for i in range(len(row)-1)])
        pred = np.array(data)
        decoded = new_model.predict(pred)
        i = input('Select Vectors to compare: ').split(' ')
        while i != '-1':
            plt.subplot(121)
            plt.imshow(decoded[int(i[0])-1].reshape(54, 54) * 255)
            plt.title(i[0])

            plt.subplot(122)
            plt.imshow(decoded[int(i[1])-1].reshape(54, 54) * 255)
            plt.title(i[1])

            plt.show()
            i = input('Select Vectors to compare: ').split(' ')


all_teams = ['BOS', 'LAC', 'CHI', 'PHI', 'CLE', 'MEM', 'ATL', 'SAC', 'UTA', 'POR', 'MIN', 'OKC', 'TOR', 'BKN', 'NOP',
             'GSW', 'MIA', 'DAL', 'PHX', 'IND', 'SAS', 'MIL', 'DET', 'NYK', 'CHA', 'WAS', 'HOU', 'LAL', 'DEN', 'ORL']

groups = {'BOS': 1, 'LAC': 1, 'CHI': 1, 'PHI': 1, 'CLE': 1, 'MEM': 1, 'ATL': 1, 'SAC': 1, 'UTA': 1, 'POR': 1,
          'MIN': 1, 'OKC': 1, 'TOR': 1, 'BKN': 1, 'NOP': 1, 'GSW': 1, 'MIA': 1, 'DAL': 1, 'PHX': 1, 'IND': 1,
          'SAS': 1, 'MIL': 1, 'DET': 1, 'NYK': 1, 'CHA': 1, 'WAS': 1, 'HOU': 1, 'LAL': 1, 'DEN': 1, 'ORL': 1}


class TeamSimilarity:
    def __init__(self, grouped=False):
        self.team_dict = dict()
        for i in range(30):
            self.team_dict[all_teams[i]] = i
        self.model = None
        self.x_train = None
        self.x_test = None
        self.y_train = None
        self.y_test = None
        self.get_data()

    def get_data(self, grouped=False):
        x = list()
        y = list()
        with open('image_autoencoder_vectors.csv') as file:
            reader = csv.reader(file, delimiter=',')
            for row in reader:
                x.append([float(row[i]) for i in range(len(row) - 1)])
                if grouped:
                    y.append(groups[row[-1]])
                else:
                    y.append(self.team_dict[row[-1]])
        num_of_groups = 3 if grouped else 30
        self.x_train = np.array(x[:int(len(y) * 0.95)])
        self.y_train = to_categorical(np.array(y[:int(len(y) * 0.95)]), num_of_groups)
        self.x_test = np.array(x[int(len(y) * 0.95):])
        self.y_test = to_categorical(np.array(y[int(len(y) * 0.95):]), num_of_groups)

    def create_model(self):
        self.model = Sequential()
        self.model.add(Dense(30, activation='softmax', input_dim=72))
        self.model.compile(loss='categorical_crossentropy', optimizer='sgd', metrics=['accuracy'])
        self.model.summary()

    def fit_model(self):
        tensorboard = TensorBoard(log_dir="logs/{}_weights".format(time()))
        checkpointer = ModelCheckpoint(filepath='../match_data/weight_model.h5', verbose=1,
                                       save_best_only=True)
        self.model.fit(self.x_train, self.y_train, epochs=400, batch_size=128,
                       validation_data=(self.x_test, self.y_test), callbacks=[tensorboard, checkpointer])

    def get_weights(self):
        self.model = load_model('../match_data/weight_model.h5')
        self.model.load_weights('../match_data/weight_model.h5')
        p = self.model.predict(self.x_train)
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


if __name__ == '__main__':
    ia = ImageAutoencoder()
    #ia.create_model()
    ia.load_model()
    ia.show_attack('../match_data/image_autoencoder_vectors_hc.csv')
    #ia.make_data_from_network()
    #ia.fit_model()
    #ia.predict()
    ia.show_kernels()
    #ts = TeamSimilarity()
    #ts.create_model()
    #ts.fit_model()
    #ts.get_weights()
