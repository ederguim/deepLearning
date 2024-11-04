import keras
import numpy as np
import tensorflow as tf
import sklearn

from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import InputLayer, Dense, Flatten, Conv2D, MaxPooling2D
from tensorflow.keras import utils as np_utils
import numpy as np
from sklearn.model_selection import StratifiedKFold

(x, y), (x_teste, y_teste) = mnist.load_data()
x = x.reshape(x.shape[0], 28, 28, 1)
x = x.astype('float32')
x /= 255
y = np_utils.to_categorical(y, 10)

seed = 5
np.random.seed(seed)
kfold = StratifiedKFold(n_splits = 5, shuffle=True, random_state = seed)

resultados = []
for indice_treinamento, indice_teste in kfold.split(x, np.zeros(shape = (y.shape[0], 1))):
    print("Índices treinamento: ", indice_treinamento, "Índice teste: ", indice_teste)
    classificador = Sequential()
    classificador.add(InputLayer(shape=(28, 28, 1)))
    classificador.add(Conv2D(32, (3, 3), activation='relu'))
    classificador.add(MaxPooling2D(pool_size=(2,2)))
    classificador.add(Flatten())
    classificador.add(Dense(units=128, activation='relu'))
    classificador.add(Dense(units=10, activation='softmax'))
    classificador.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    classificador.fit(x[indice_treinamento], y[indice_treinamento], batch_size = 128, epochs = 5)
    precisao = classificador.evaluate(x[indice_teste], y[indice_teste])
    resultados.append(precisao[1])

print(np.array(resultados).mean())
print(np.array(resultados).std())