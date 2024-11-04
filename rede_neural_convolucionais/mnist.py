import tensorflow as tf
import keras
import matplotlib
import numpy as np

from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import InputLayer, Dense, Flatten, Dropout, Conv2D, MaxPooling2D, BatchNormalization
from tensorflow.keras import utils as np_utils
import matplotlib.pyplot as plt

(x_treinamento, y_treinamento), (x_teste, y_teste) = mnist.load_data()
plt.imshow(x_treinamento[1], cmap='gray')
plt.title('Classe ' + str(y_treinamento[1]))

x_treinamento = x_treinamento.reshape(x_treinamento.shape[0], 28, 28, 1)
x_teste = x_teste.reshape(x_teste.shape[0], 28, 28, 1)
print(x_treinamento.shape, x_teste.shape)

x_treinamento = x_treinamento.astype('float32')
x_teste = x_teste.astype('float32')

x_treinamento /= 255
x_teste /= 255

print(x_treinamento.max(), x_treinamento.min())
print(x_teste.max(), x_teste.min())

# [5, 0, 4] => [1., 0., 0.]
y_treinamento = np_utils.to_categorical(y_treinamento, 10)
y_teste = np_utils.to_categorical(y_teste, 10)

rede_neural = Sequential()

rede_neural.add(InputLayer(shape=(28, 28, 1)))
rede_neural.add(Conv2D(filters=32, kernel_size=(3,3), activation='relu'))
rede_neural.add(BatchNormalization())

rede_neural.add(MaxPooling2D(pool_size=(2,2)))
rede_neural.add(Conv2D(filters=32, kernel_size=(3,3), activation='relu'))
rede_neural.add(BatchNormalization())
rede_neural.add(MaxPooling2D(pool_size=(2,2)))

rede_neural.add(Flatten()) # 13 * 13 * 32
rede_neural.add(Dense(units=128, activation='relu'))
rede_neural.add(Dropout(0.2))
rede_neural.add(Dense(units=128, activation='relu'))
rede_neural.add(Dropout(0.2))
rede_neural.add(Dense(units=10, activation='softmax'))
rede_neural.summary()

rede_neural.compile(loss = 'categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
rede_neural.fit(x_treinamento, y_treinamento, batch_size=128, epochs=5, validation_data=(x_teste, y_teste))
resultado = rede_neural.evaluate(x_teste, y_teste)