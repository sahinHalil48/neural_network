'''
A multi-layer neural network, built by chaning together Layer objects
'''

from layer import Layer 

class Network:

    def __init__(self, layer_sizes):
        # initialize the network by creating a Layer for each pair of consecutive sizes in layer_sizes

        self.layers = list()

        for i in range(len(layer_sizes) - 1):
            layer = Layer(num_inputs=layer_sizes[i],
                          num_neurons=layer_sizes[i+1])

            self.layers.append(layer)


    def forward(self,inputs):

        previous_output = inputs

        for layer in self.layers:
            current_output = layer.forward(previous_output)

        return current_output

    def backward(self,delta,learning_rate):

        for layer in reversed(self.layers):
            delta = layer.backward(delta, learning_rate)





        