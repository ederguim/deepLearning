import pandas as pd
import tensorflow as tf
import sklearn

from tensorflow.keras.layers import Dense, Dropout, Activation, Input
from tensorflow.keras.models import Model
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.metrics import mean_absolute_error

data = pd.read_csv('arquivos/games.csv')

data = data.drop('Other_Sales', axis=1)
data = data.drop('Global_Sales', axis=1)
data = data.drop('Developer', axis=1)

print(data.shape)
print(data.isnull().sum())

# Drop todos os valores nulos da base
data = data.dropna(axis=0)
print(data.shape)
print(data.isnull().sum())

# Drop a coluna name
print(data['Name'].value_counts())
data = data.drop('Name', axis=1)
print(data.shape)

print(data.columns)
x = data.iloc[:, [0, 1, 2, 3, 7, 8, 9, 10, 11]].values


y_na = data.iloc[:, 4].values
y_eu = data.iloc[:, 5].values
y_jp = data.iloc[:, 6].values

print(data['Platform'].value_counts())
print(data.columns)
oneHotEncoder = ColumnTransformer(transformers=[('OneHot', OneHotEncoder(), [0, 2, 3, 8])], remainder='passthrough')
x = oneHotEncoder.fit_transform(x).toarray()
print(x.shape)

# camada oculta = (entrada + saida) / 2 
camada_entrada = Input(shape = (303,))
camada_oculta1 = Dense(units = 153, activation='relu')(camada_entrada)
camada_oculta2 = Dense(units = 153, activation='relu')(camada_oculta1)
camada_saida1 = Dense(units = 1, activation='linear')(camada_oculta2)
camada_saida2 = Dense(units = 1, activation='linear')(camada_oculta2)
camada_saida3 = Dense(units = 1, activation='linear')(camada_oculta2)

regressor = Model(inputs=camada_entrada, outputs=[camada_saida1, camada_saida2, camada_saida3])
regressor.compile(optimizer='adam', loss='mse')
regressor.fit(x, [y_na, y_eu, y_jp], epochs=500, batch_size=100)
previsao_na, previsao_eu, previsao_jp = regressor.predict(x)

print(previsao_na, previsao_na.mean())
print(y_na, y_na.mean())

print(previsao_eu, previsao_eu.mean())
print(y_eu, y_eu.mean())

print(previsao_jp, previsao_jp.mean())
print(y_jp, y_jp.mean())

# Previsao de erro
print(mean_absolute_error(y_na, previsao_na))
print(mean_absolute_error(y_eu, previsao_eu))
print(mean_absolute_error(y_jp, previsao_jp))









