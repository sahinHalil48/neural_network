import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split


from loss import (
    mean_squared_error,
    mean_squared_error_derivative,
    one_hot_encode,
    sigmoid_derivative
)

from network import Network

def main():

    # data uploading ... 

    mnist = fetch_openml('mnist_784', version=1, cache=True, as_frame=False)
    
    X = mnist["data"]
    y = mnist["target"].astype(int)
    
    # Normalization: 0-255 pixels squeshed into range 0-1 
    X = X / 255.0
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=20000, test_size=4000, random_state=42)
    
    num_classes = 10
    epochs = 30
    learning_rate = 0.5

    print(f"train data: {X_train.shape[0]} times | Test data: {X_test.shape[0]} times")

    layer_sizes = [784, 64, num_classes]
    network = Network(layer_sizes)

    loss_history = []  

    # TRAINING

    for epoch in range(epochs):

        epoch_loss = 0.0 

        for i in range(len(X_train)):

            sample_input = X_train[i].reshape(1, -1)
            target_vector = one_hot_encode(y_train[i], num_classes).reshape(1, -1)

            predictions = network.forward(sample_input)

            loss = mean_squared_error(predictions,target_vector)
            epoch_loss += loss

            initial_delta = mean_squared_error_derivative(predictions,target_vector) * sigmoid_derivative(predictions)

            network.backward(initial_delta , learning_rate)

        avg_loss = epoch_loss / len(X_train)
        loss_history.append(avg_loss)
        print(f"Epoch {epoch + 1:2d} / {epochs} | MSE: {avg_loss:.4f}")




    correct_predictions = 0
    
    for i in range(len(X_test)):
        sample_input = X_test[i].reshape(1, -1)
        pred_vector = network.forward(sample_input)
        pred_class = np.argmax(pred_vector)
        
        if pred_class == y_test[i]:
            correct_predictions += 1
            
    accuracy = correct_predictions / len(X_test)
    print(f"Test Seti Doğruluk Oranı (Accuracy): %{accuracy * 100:.2f}")

if __name__ == "__main__":
    main()
