# XOR Neural Network

Originally, I was attempting to create this neural network from scratch using a
class Unit that would contain the units value, next & prev layers and the
weights from the previous nodes. Doing so made the process way more convoluted
than if I were to use matrices and vectors.

I found a [tutorial](https://maviccprp.github.io/a-neural-network-from-scratch-in-just-a-few-lines-of-python-code/)
online which really helped me understand why matrices are so important for 
neural networks.

After following the guide and making some adjustments to the network to make it
my own, I have a successful XOR network! My goal now is to make it work with two
output nodes, one representing 1 and the other 0. Then, I can apply the same
concepts used here for my next deep learning exercise; a neural network w/ out 
machine learning libraries that works on the MNIST data set.
