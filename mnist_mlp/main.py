# Copyright Austin Ward 2018


from network import Network
import numpy as np
import mnist


def format_data(images, labels, vector_y=True):
    X = images 
    xs = X.reshape((X.shape[0], X.shape[1] * X.shape[2]))
    xs = xs/255
    ys = labels 
    if vector_y:
        shape = (len(xs), 10)
        vectorized_y = np.zeros(shape)
        for idx, label in enumerate(ys):
            vectorized_y[idx][label] = 1

        data = []
        for x, y in zip(xs, vectorized_y):
            data.append((x.reshape(-1, 1).astype(float), y.reshape(-1, 1)))
    else:
        data = []
        for x, y in zip(xs, ys):
            data.append((x.reshape(-1, 1).astype(float), y.reshape(-1, 1)))

    return data

train = mnist.train_images()[:-10000], mnist.train_labels()[:-10000]
validation = mnist.train_images()[-10000:], mnist.train_labels()[-10000:]
test  = mnist.test_images(), mnist.test_labels()

train = format_data(*train)
validation = format_data(*validation, vector_y=False)
test = format_data(*test, vector_y=False)

# Build the network
net = Network([784, 50, 50, 10])

# Train the network
net.SGD(train, 30, 10, 3, validation)

# Test the network
print("Test data: {:3.2f}% correct".format(
    100 * net.evaluate(test)/len(test)
    ))
