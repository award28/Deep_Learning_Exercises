# Copyright Austin Ward 2018


import numpy as np
import random


def sigmoid(z, derive=False):
    if derive:
        return sigmoid(z) * (1 - sigmoid(z))
    return 1.0/(1.0 + np.exp(-z))


class Network(object):
    def __init__(self, sizes):
        self.num_layers = len(sizes)

        # List of sizes for input, hidden and output layers
        self.sizes = sizes

        # Create bias for each weight
        self.biases = [np.random.randn(y, 1) for y in sizes[1:]]

        # Create weights going from x->y with shape (y, x)
        self.weights = [np.random.randn(y, x) 
                for x, y in zip(sizes[:-1], sizes[1:])]


    def feedforward(self, a):
        for b, w in zip(self.biases, self.weights):
            a = sigmoid(np.dot(w, a) + b)
        return a


    # Stochastic Gradient Descent
    def SGD(self, train_data, epochs, batch_size, l_rate, validation_data=None):
        n = len(train_data)
        for j in range(1, epochs + 1):

            # Shuffle training data on each epoch
            random.shuffle(train_data)

            # Create list of batches
            batches = [
                    train_data[k:k + batch_size]
                    for k in range(0, n, batch_size)
                    ]

            # Update biases and weights based on batches and learning rate
            for batch in batches:
                self.update_batch(batch, l_rate)

            if validation_data:
                print("Epoch {:2}: {:3.2f}%".format(
                    j, 
                    100 * self.evaluate(validation_data)/len(validation_data)
                    ))
            else:
                print("Epoch {}".format(j))


    def update_batch(self, batch, l_rate):
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]

        # For each training example in batch, add the change to the gradient
        for x, y in batch:
            delta_nabla_b, delta_nabla_w = self.backprop(x, y)
            nabla_b = [nb + dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]
            nabla_w = [nw + dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]
        
        # Update weights and biases
        self.weights = [w - (l_rate/len(batch)) * nw
                for w, nw in zip(self.weights, nabla_w)] 

        self.biases = [b - (l_rate/len(batch)) * nb
                for b, nb in zip(self.biases, nabla_b)]


    def backprop(self, x, y):
        # Init lists of gradient for cost function for biases and weights
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]

        # Save activations and z vectors for backprop
        activations = [x]
        zs = []

        # Feedforward
        for b, w in zip(self.biases, self.weights):
            zs.append(np.dot(w, activations[-1]) + b)
            activations.append(sigmoid(zs[-1]))

        # Backprop
        cost_derivative = activations[-1] - y

        # Change to propagate backwards
        delta = cost_derivative * sigmoid(zs[-1], derive=True)
        nabla_b[-1] = delta
        nabla_w[-1] = np.dot(delta, activations[-2].T)

        # Work backwards through all layers, applying changes
        for l in range(2, self.num_layers):
            sp = sigmoid(zs[-l], derive=True)
            delta = np.dot(self.weights[-l+1].T, delta) * sp
            nabla_b[-l] = delta
            nabla_w[-l] = np.dot(delta, activations[-l-1].T)

        # Return gradient
        return (nabla_b, nabla_w)


    def evaluate(self, data):
        # For each image classify and compare network output to correct output
        return sum(int(np.argmax(self.feedforward(x)) == y) for x, y in data)
