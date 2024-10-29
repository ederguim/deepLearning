import pandas as pd
import tensorflow as tf
import sklearn
import scikeras

from scikeras.wrappers import KerasClassifier
from sklearn.model_selection import cross_val_score
from tensorflow.keras.models import Sequential
from tensorflow.keras import backend as k

x = pd.read_csv('arquivos\entradas_breast.csv')
y = pd.read_csv('arquivos\saidas_breast.csv')

def criar_rede_neural():
    k.clear_session()
    rede_neural = Sequential([
        tf.keras.layers.InputLayer(shape = (30,)),
        tf.keras.layers.Dense(units = 16, activation = 'relu', kernel_initializer = 'random_uniform'),
        tf.keras.layers.Dropout(rate = 0.2),
        tf.keras.layers.Dense(units = 16, activation = 'relu', kernel_initializer = 'random_uniform'),
        tf.keras.layers.Dropout(rate = 0.2),
        tf.keras.layers.Dense(units = 1, activation = 'sigmoid'),
    ])
    otimizador = tf.keras.optimizers.Adam(learning_rate = 0.001, clipvalue = 0.5)
    rede_neural.compile(optimizer = otimizador, loss = 'binary_crossentropy', metrics = ['binary_accuracy'])
    return rede_neural

rede_neural = KerasClassifier(model=criar_rede_neural, epochs=100, batch_size=10)
resultados = cross_val_score(estimator=rede_neural, X=x, y=y, cv=10, scoring='accuracy')

print(resultados.mean())
print(resultados.std())
