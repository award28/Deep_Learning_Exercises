# Copyright Austin Ward 2018


from random import random
from math import exp

class Unit(object):
    def __init__(self, v):
        self.val = v


    def prev_layer(self, prev):
        self.prev = prev
        self.weights = [random(), random()]


    def next_layer(self, next):
        self.next = next


def relu(s):
    return s if s > 0 else 0

def sigmoid(s):
    return 1/(1 + exp(-s))

def run_network(in_, hidden, out):
    ws = hidden[0].weights
    hidden[0].val = relu(sum([i.val*w for i, w in zip(in_, ws)]))
    ws = hidden[1].weights
    hidden[1].val = relu(sum([i.val*w for i, w in zip(in_, ws)]))

    ws = out.weights
    sig = sigmoid(sum([h.val*w for h, w in zip(hidden, ws)]))
    out.val = int(sig > 0.5)


# Creating input, hidden and output layer with following shape
'''
O---->O
 \   ^ \
  \ /   v
   X     O
  / \   ^
 /   v /
O---->O
'''

i_layer = [Unit(0), Unit(0)]
h_layer = [Unit(0), Unit(0)]
o_layer = [Unit(0)]

o_layer[0].prev_layer(h_layer)
for i, h in zip(i_layer, h_layer):
    i.next_layer(h_layer)
    h.prev_layer(i_layer)
    h.next_layer(o_layer)

features = [
        [0, 0],
        [0, 1],
        [1, 0],
        [1, 1]
        ]
labels = [
        0,
        1,
        1,
        0
        ]

for f, l in zip(features, labels):
    i_layer[0].val = f[0]
    i_layer[1].val = f[1]
    run_network(i_layer, h_layer, o_layer[0])
    print("Correct label: " + str(l))
    print("Predicted label: " + str(o_layer[0].val))
