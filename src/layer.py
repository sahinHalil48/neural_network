# A layer of multi neurons which each receive same inputs and each produce one output. 

import numpy as np 

from neuron import sigmoid 

class Layer:

    def __init__(self, num_inputs, num_neurons):
        # initialize the layer with random weights and zero biases

        self.weights = np.random.randn(num_inputs , num_neurons)

        self.biases = np.zeros(num_neurons)



    def forward(self,inputs):
        # compute the layer's output for a given input vector. 

        self.inputs = inputs 

        z = np.dot(inputs,self.weights) + self.biases

        self.output = sigmoid(z)

        return self.output


    def backward(self, delta, learning_rate):
        #perform the backward pass: compute gradients and update weights and biases using gradient descent 

        # weight_gradient = d loss / dW  = (d loss / dz) *  (dz / dW)
        # bias_gradient = d loss / db = (d loss / dz) * (dz / db)(=1)
        # input_gradient = d loss / d input = (d loss / d z) * (d z / d input) 
        # delta = d loss / dz

        # shapes : inputs (N,D_in), weights (D_in,D_out) biases (1,D_out) => z,delta (N ,D_out)

        delta = delta * (self.output * (1 - self.output)) # sigmoid derivative 

        weight_gradients = np.dot(self.inputs.T, delta)

        bias_gradient = delta 

        temp_weights = self.weights 

        self.weights = self.weights - learning_rate * weight_gradients

        self.biases = self.biases - learning_rate * bias_gradient

        input_gradient = np.dot(delta,temp_weights.T)

        return input_gradient




