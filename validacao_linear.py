import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
import tensorflow as tf
from tensorflow.keras.models import Sequential

# previsores
x = pd.read_csv('arquivos\entradas_breast.csv')
y = pd.read_csv('arquivos\saidas_breast.csv')

x_treinamento, x_teste, y_treinamento, y_teste = train_test_split(x, y, test_size=0.25)

print(f" {x_treinamento.shape}, {x_teste.shape}")
print(f" {y_treinamento.shape}, {y_teste.shape}")

# 30 neuronios na camada de entrada, 16 neuronios na camada oculta, 1 neuronio na camada de saida
# camada oculta funcao relu, camada de saida funcao sigmoid
rede_neural = Sequential([
    tf.keras.layers.InputLayer(shape = (30,)),
    tf.keras.layers.Dense(units = 16, activation = 'relu', kernel_initializer = 'random_uniform'),
    tf.keras.layers.Dense(units = 16, activation = 'relu', kernel_initializer = 'random_uniform'),
    tf.keras.layers.Dense(units = 1, activation = 'sigmoid'),
])


rede_neural.summary()

# otimizador custom
otimizador = tf.keras.optimizers.Adam(learning_rate = 0.001, clipvalue = 0.5)
rede_neural.compile(optimizer = otimizador, loss = 'binary_crossentropy', metrics = ['binary_accuracy'])

# otimizador default
rede_neural.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['binary_accuracy'])
rede_neural.fit(x_treinamento, y_treinamento, batch_size = 10, epochs = 100)

previsoes = rede_neural.predict(x_teste)
previsoes = previsoes > 0.5


pesos0 = rede_neural.layers[0].get_weights()
print(pesos0)
print(len(pesos0))

pesos1 = rede_neural.layers[1].get_weights()
print(pesos1)
print(len(pesos1))


print(rede_neural.evaluate(x_teste, y_teste))
print(accuracy_score(y_teste, previsoes)) # percentual de acerto
print(confusion_matrix(y_teste, previsoes)) # matrix do resultado