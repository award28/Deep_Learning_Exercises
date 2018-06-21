# MNIST Neural Network

### Date: 6/10/18

The goal of this project is to make an MLP that can predict hand written digits
correctly. The dataset is provided [here.](http://yann.lecun.com/exdb/mnist/)

So far I've added another hidden layer to the xor network, now both have 16 
nodes. With that, I can start working on the MNIST data set with this network.

### Date: 6/21/18

After breaking the model up into sub problems and tackling those, I have a 
neural network that will now classify hand written digits correctly ~96% of the
time.

Some of the challenges I faced:
  - Overflow error on the exponential inside the sigmoid.
    - Changed the range of my values from 0-255 to 0-1, proportionally.
  - Getting the correct shape for my weights.
    - Understanding the notation for weights and applying that notation.

Thank you [Michael Nielsen](http://michaelnielsen.org/) for writting the
_[Neural Networks and Deep Learning](http://neuralnetworksanddeeplearning.com/)_
 online book, this really helped me understand neural networks and break down my
 network into sub problems.
