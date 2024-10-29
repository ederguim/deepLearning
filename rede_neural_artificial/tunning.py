import pandas as pd
import tensorflow as tf
import sklearn
import scikeras

from scikeras.wrappers import KerasClassifier
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from tensorflow.keras.models import Sequential
from tensorflow.keras import backend as k

x = pd.read_csv('arquivos\entradas_breast.csv')
y = pd.read_csv('arquivos\saidas_breast.csv')

def criar_rede_neural(optimizer, loss, kernel_initializer, activation, neurons):
    k.clear_session()
    rede_neural = Sequential([
        tf.keras.layers.InputLayer(shape = (30,)),
        tf.keras.layers.Dense(units = neurons, activation = activation, kernel_initializer = kernel_initializer),
        tf.keras.layers.Dropout(rate = 0.2),
        tf.keras.layers.Dense(units = neurons, activation = activation, kernel_initializer = kernel_initializer),
        tf.keras.layers.Dropout(rate = 0.2),
        tf.keras.layers.Dense(units = 1, activation = 'sigmoid'),
    ])
    rede_neural.compile(optimizer = optimizer, loss = loss, metrics = ['binary_accuracy'])
    return rede_neural

rede_neural = KerasClassifier(model=criar_rede_neural)


parametros = {
    'batch_size': [10, 30],
    'epochs': [50, 100],
    'model__optimizer': ['adam', 'sgd'],
    'model__loss': ['binary_crossentropy', 'hinge'],
    'model__kernel_initializer': ['random_uniform', 'normal'],
    'model__activations': ['relu', 'tanh'],
    'model__neurons': [16, 8],
}

_parametros = {
    'batch_size': [10, 30],
    'epochs': [50],
    'model__optimizer': ['adam'],
    'model__loss': ['binary_crossentropy'],
    'model__kernel_initializer': ['random_uniform', 'normal'],
    'model__activations': ['relu'],
    'model__neurons': [16],
}

__parametros = {
    'batch_size': [10, 30],
    'epochs': [50],
    'model__optimizer': ['sgd'],
    'model__loss': ['hinge'],
    'model__kernel_initializer': ['normal'],
    'model__activations': ['tanh'],
    'model__neurons': [16],
}

grid_search = GridSearchCV(estimator=rede_neural, param_grid=parametros, scoring='accuracy', cv=10)
grid_search = grid_search.fit(x,y)

melhores_parametros = grid_search.best_params_
print(melhores_parametros)

melhor_precisao = grid_search.best_score_
print(melhor_precisao)

