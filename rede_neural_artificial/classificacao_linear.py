import pandas as pd
import numpy as np
import tensorflow as tf
import sklearn

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras import utils as np_utils
from tensorflow.keras import backend as k
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score

data = pd.read_csv('arquivos/iris.csv')
x = data.iloc[:, 0:4].values
y = data.iloc[:, 4].values

encoder = LabelEncoder()
# convert tipo string em integer [0, 1, 2]
y = encoder.fit_transform(y)
# classifica das categorias [1., 0., 0.], [0., 1., 0.], [0., 0., 1.] 
y = np_utils.to_categorical(y)
x_treinamento, x_teste, y_treinamento, y_teste = train_test_split(x, y, test_size=0.25)
print(f"{x_treinamento.shape}, {x_teste.shape}")
print(f"{y_treinamento.shape}, {y_teste.shape}")

rede_neural = Sequential([
    tf.keras.layers.InputLayer(shape = (4,)),
    tf.keras.layers.Dense(units=4, activation='relu'),
    tf.keras.layers.Dense(units=4, activation='relu'),
    tf.keras.layers.Dense(units=3, activation='softmax')
])

rede_neural.summary()
rede_neural.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['categorical_accuracy'])
rede_neural.fit(x_treinamento, y_treinamento, batch_size=10, epochs=100)
rede_neural.evaluate(x_teste, y_teste)
previsoes = rede_neural.predict(x_teste)
previsoes = previsoes > 0.5
print(previsoes)

y_teste2 = [np.argmax(t) for t in y_teste]
print(y_teste2)

previsoes2 = [np.argmax(t) for t in previsoes]
print(previsoes2)

print(accuracy_score(y_teste2, previsoes2))
print(confusion_matrix(y_teste2, previsoes2))
