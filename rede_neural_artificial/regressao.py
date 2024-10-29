import pandas as pd
import tensorflow as tf
import sklearn

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer

data = pd.read_csv('arquivos/autos.csv', encoding='ISO-8859-1')

print(data['vehicleType'].value_counts())
print(data['gearbox'].value_counts())
print(data['model'].value_counts())
print(data['fuelType'].value_counts())
print(data['notRepairedDamage'].value_counts())

# drop colunas desnecessarias
data = data.drop('dateCrawled', axis=1)
data = data.drop('dateCreated', axis=1)
data = data.drop('nrOfPictures', axis=1)
data = data.drop('postalCode', axis=1)
data = data.drop('lastSeen', axis=1)
data = data.drop('name', axis=1)
data = data.drop('seller', axis=1)
data = data.drop('offerType', axis=1)

# identificando valores invalidos
data.loc[data['price'] <= 10]
data.loc[data['price'] > 350000]
data.loc[pd.isnull(data['vehicleType'])]
data.loc[pd.isnull(data['gearbox'])]
data.loc[pd.isnull(data['model'])]
data.loc[pd.isnull(data['fuelType'])]
data.loc[pd.isnull(data['notRepairedDamage'])]

# atualizar valores nulos
valores = {'vehicleType': 'limousine',
           'gearbox': 'manuell',
           'model': 'golf',
           'fuelType': 'benzin',
           'notRepairedDamage': 'nein',
}

data = data.fillna(value=valores)

# filtrando valores invalidos
data = data[data['price'] < 350000]
data = data[data['price'] > 10]
print(data.shape)

# verifica valores nulos
print(data.isnull().sum())
print(data.columns)

x = data.iloc[:, 1:12].values
y = data.iloc[:, 0].values

# Encoder
print(data['brand'].value_counts())
oneHotEncoder = ColumnTransformer(transformers=[("OneHot", OneHotEncoder(), [0, 1, 3, 5, 8, 9, 10])], remainder='passthrough')
x = oneHotEncoder.fit_transform(x).toarray()
print(x)
print(x.shape)

# (316 + 1) / 2 => definir neuronios ocultos
regressor = Sequential([
    tf.keras.layers.InputLayer(shape = (316,)),
    tf.keras.layers.Dense(units=158, activation='relu'),
    tf.keras.layers.Dense(units=158, activation='relu'),
    tf.keras.layers.Dense(units=1, activation='linear')
])

regressor.summary()
regressor.compile(loss='mean_absolute_error', optimizer='adam', metrics=['mean_absolute_error'])
regressor.fit(x, y, batch_size=300, epochs=10)
previsoes = regressor.predict(x)
print(previsoes)
print(y)
print(y.mean())
print(previsoes.mean())




