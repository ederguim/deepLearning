import keras
import tensorflow as tf

from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import InputLayer, Dense, Flatten, Conv2D, MaxPooling2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import utils as np_utils

(x_treinamento, y_treinamento), (x_teste, y_teste) = mnist.load_data()

x_treinamento = x_treinamento.reshape(x_treinamento.shape[0], 28, 28, 1)
x_teste = x_teste.reshape(x_teste.shape[0], 28, 28, 1)
x_treinamento = x_treinamento.astype('float32')
x_teste = x_teste.astype('float32')
x_treinamento /= 255
x_teste /= 255
y_treinamento = np_utils.to_categorical(y_treinamento, 10)
y_teste = np_utils.to_categorical(y_teste, 10)

classificador = Sequential()
classificador.add(InputLayer(shape=(28, 28, 1)))
classificador.add(Conv2D(32, (3, 3), activation='relu'))
classificador.add(MaxPooling2D(pool_size=(2, 2)))
classificador.add(Flatten())
classificador.add(Dense(units=128, activation='relu'))
classificador.add(Dense(units=10, activation='softmax'))
classificador.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

gerador_treinamento = ImageDataGenerator(rotation_range=7, horizontal_flip=True,
                                         shear_range=0.2, height_shift_range=0.07,
                                         zoom_range=0.2)

gerador_teste = ImageDataGenerator()
base_treinamento = gerador_treinamento.flow(x_treinamento, y_treinamento, batch_size = 128)
base_teste = gerador_teste.flow(x_teste, y_teste, batch_size = 128)
classificador.fit(base_treinamento, epochs=5, validation_data=base_teste)
