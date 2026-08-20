import numpy as np
def mean_squared_error (predictions,targets):
    #Compute the Mean Squared Error between predictions and targets.

    difference = predictions - targets 

    result = np.mean(difference**2)

    return result 


def one_hot_encode(label , num_classes):
    # convert a single integer label into a one hot encoded vector 

    one_hot_vector = np.zeros(num_classes)

    for i in range(num_classes):
        if i == label :
            one_hot_vector[i] = 1
        else:
            pass

    return one_hot_vector



def sigmoid_derivative(sigmoid_output):

    # σ'(z) = σ(z) × (1 - σ(z))

    result = sigmoid_output * ( 1 - sigmoid_output)

    return result



def mean_squared_error_derivative(predictions,targets):
    # d(Loss)/d(prediction_i) = (2/n) × (prediction_i - target_i)

    result = 2 / len(predictions) * (predictions - targets)

    return result 
    


