import pandas as pd
import tensorflow as tf
import sklearn
import scikeras

import time
from scikeras.wrappers import KerasRegressor
from tensorflow.keras import backend as k
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn import metrics

inicio = time.time()

data = pd.read_csv('autos.csv', encoding='ISO-8859-1')

data = data.drop('dateCrawled', axis=1)
data = data.drop('dateCreated', axis=1)
data = data.drop('nrOfPictures', axis=1)
data = data.drop('postalCode', axis=1)
data = data.drop('lastSeen', axis=1)
data = data.drop('name', axis=1)
data = data.drop('seller', axis=1)
data = data.drop('offerType', axis=1)

data = data[data.price > 10]
data = data.loc[data.price < 350000]

valores = {'vehicleType': 'limousine',
           'gearbox': 'manuell',
           'model': 'golf',
           'fuelType': 'benzin',
           'notRepairedDamage': 'nein'}
data = data.fillna(value=valores)

X = data.iloc[:, 1:12].values
y = data.iloc[:, 0].values

onehotencoder = ColumnTransformer(transformers=[("OneHot", OneHotEncoder(), [0, 1, 3, 5, 8, 9, 10])], remainder='passthrough')
X = onehotencoder.fit_transform(X).toarray()

X.shape

def criar_rede():
    k.clear_session()
    regressor = Sequential([
        tf.keras.layers.InputLayer(shape=(316,)),
        tf.keras.layers.Dense(units=158, activation='relu'),
        tf.keras.layers.Dense(units=158, activation='relu'),
        tf.keras.layers.Dense(units=1, activation='linear')])
    regressor.compile(loss='mean_absolute_error', optimizer='adam', metrics=['mean_absolute_error'])
    return regressor

regressor = KerasRegressor(model = criar_rede, epochs = 100, batch_size = 300)

resultados = cross_val_score(estimator = regressor, X = X, y = y, cv = 5, scoring = 'neg_mean_absolute_error')

fim = time.time()

(fim - inicio) / 60 / 60

abs(resultados)

abs(resultados.mean())

resultados.std()