# Copyright Austin Ward 2018


import numpy as np


'''
Creating neural network with the following shape

O---->O
 \   ^ \ 
  \ /   v
   X     O
  / \   ^
 /   v /
O---->O
'''


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
        [0],
        [1],
        [1],
        [0]
        ])

hidden_units = 2
w01 = np.random.random((len(features[0]), hidden_units))
w12 = np.random.random((hidden_units, 1))

eta = 0.1
epochs = range(1)

for _ in epochs:
    # Feed forward
    z_h = np.dot(features, w01)
    a_h = sigmoid(z_h)

    z_o = np.dot(a_h, w12)
    a_o = sigmoid(z_o)

    # Squared error function
    a_o_err = (1/2) * np.power(a_o - labels, 2)

    # Backprop
    # Output layer
    delta_a_o_err = a_o - labels
    delta_z_o = sigmoid(a_o, derive=True)
    delta_w12 = a_h
    delta_output = np.dot(delta_w12.T, (delta_a_o_err * delta_z_o))

    # Hidden layer
    delta_a_h = np.dot(delta_a_o_err * delta_z_o, w12.T)
    delta_z_h = sigmoid(a_h, derive=True)
    delta_w01 = features
    delta_hidden = np.dot(delta_w01.T, delta_a_h*delta_z_h)

    # Update weights
    w01 -= (eta * delta_hidden)
    w12 -= (eta * delta_output)


z_h = np.dot(features, w01)
a_h = sigmoid(z_h)

z_o = np.dot(a_h, w12)
a_o = sigmoid(z_o)

print(a_o)

for o, l in zip(a_o, labels):
    prediction = int(o > .5)
    print("Label: " + str(l))
    print("Prediction: " + str(prediction))
