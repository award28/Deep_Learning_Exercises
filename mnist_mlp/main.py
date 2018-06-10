# Copyright Austin Ward 2018


import numpy as np


def relu(z, derive=False):
    if derive:
        z[ z > 0] = 1
    return np.maximum(z, 0, z)


def sigmoid(z, derive=False):
    if derive:
        return z * (1 - z)
    return 1/(1 + np.exp(-z))


features = np.array([
        [0, 0],
        [0, 1],
        [1, 0],
        [1, 1]
        ])
labels = np.array([
        [1, 0],
        [0, 1],
        [0, 1],
        [1, 0]
        ])

hidden_units = 16
output_units = 2

w01 = np.random.random((len(features[0]), hidden_units))
w12 = np.random.random((hidden_units, hidden_units))
w23 = np.random.random((hidden_units, output_units))

# learning rate
eta = 0.1
# change to 20000 for training
epochs = range(20000)

for _ in epochs:
    # Feed forward
    z_h1 = np.dot(features, w01)
    a_h1 = sigmoid(z_h1)

    z_h2 = np.dot(a_h1, w12)
    a_h2 = sigmoid(z_h2)

    z_o = np.dot(a_h2, w23)
    a_o = sigmoid(z_o)

    # Squared error function
    a_o_err = (1/2) * np.power(a_o - labels, 2)

    # Backprop
    # Output layer
    delta_a_o_err = a_o - labels
    delta_z_o = sigmoid(a_o, derive=True)
    delta_w23 = a_h2
    delta_output = np.dot(delta_w23.T, delta_a_o_err * delta_z_o)

    # Hidden layers
    delta_a_h2 = np.dot(delta_a_o_err * delta_z_o, w23.T)
    delta_z_h2 = sigmoid(a_h2, derive=True)
    delta_w12 = a_h1
    delta_hidden2 = np.dot(delta_w12.T, delta_a_h2 * delta_z_h2)

    delta_a_h1 = np.dot(delta_a_h2, w12.T)
    delta_z_h1 = sigmoid(a_h1, derive=True)
    delta_w01 = features
    delta_hidden1 = np.dot(delta_w01.T, delta_a_h1 * delta_z_h1)

    # Update weights
    w01 -= (eta * delta_hidden1)
    w12 -= (eta * delta_hidden2)
    w23 -= (eta * delta_output)

z_h1 = np.dot(features, w01)
a_h1 = sigmoid(z_h1)

z_h2 = np.dot(a_h1, w12)
a_h2 = sigmoid(z_h2)

z_o = np.dot(a_h2, w23)
a_o = sigmoid(z_o)

print(a_o)

for o, l in zip(a_o, labels):
    print("Label: " + str(l))
    print("Prediction: " + str(o))
