import numpy as np 

def ReLU(z):
  return np.maximum(0,z)

def linear(z):
  return z

def linearPrime(z):
  return 1

def ReLUPrime(z):
  return z>0

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def sigmoidPrime(z):
    sig = sigmoid(z)
    return sig * (1 - sig)

def softmax(z):
    e_z = np.exp(z - np.max(z, axis=0))
    return e_z / np.sum(e_z, axis=0)

def softmaxPrime(z):
    return 1

def tanh(z):
    return np.tanh(z)

def tanhPrime(z):
    return 1 - np.tanh(z) ** 2
