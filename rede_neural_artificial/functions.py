import numpy as np

# Setp
def setp_function(soma):
    if soma >= 1:
        return 1
    return 0
print(f"Setp -10 => {setp_function(-10)}")
print(f"Setp 10 => {setp_function(10)}")

# Sigmoid
def sigmoid_function(soma):
    return 1 / (1 + np.exp(-soma))
print(f"Sigmoid 10 => {sigmoid_function(10)}")
print(f"Sigmoid 1 => {sigmoid_function(1)}")


# Hyperbolic
def tahn_function(soma):
    return (np.exp(soma) - np.exp(-soma)) / (np.exp(soma) + np.exp(-soma))
print(f"Hyperbolic -10 => {tahn_function(-10)}")
print(f"Hyperbolic 10 => {tahn_function(10)}")
print(f"Hyperbolic 1 => {tahn_function(1)}")

# ReLu
def reLu_function(soma):
    if soma >= 0:
        return soma
    return 0
print(f"ReLu -10 => {reLu_function(-10)}")
print(f"ReLu 10 => {reLu_function(10)}")

# Softmax
def softmax_function(x):
    ex = np.exp(x)
    return ex / ex.sum()
print(f"Softmax [7.0, 2.0, 1.3] => {softmax_function([7.0, 2.0, 1.3])}")
print(f"Softmax [7.0, 7.0, 5.3] => {softmax_function([7.0, 7.0, 5.3])}")

# Linear
def linear_function(soma):
    return soma
print(f"Linear 10 => {linear_function(10)}")



