# a single neuron 

import numpy as np 

def sigmoid(z):
    # sigmoid function squeshes any real numbor into the range (0,1).
    # for z = 0 sigmoid = 0.5 , for z aproaches to positive infinity sigmoid aproaches to 1. 

    output = 1 / (1 + np.exp(-1*z))

    return output


class Neuron:

    # A single neuron with weights, a bias and a sigmoid activation. 

    def __init__(self, num_inputs):
        # initialize a neuron with zero bias and random weights. 

        self.weights = np.random.randn(num_inputs)
        self.bias = 0 


    def forward(self, inputs ):
        # compute a neuron's output for a given input 
        # output = input dot product weights + bias 

        z = np.dot(inputs,self.weights) + self.bias

        return sigmoid(z)