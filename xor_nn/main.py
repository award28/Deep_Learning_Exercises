# Copyright Austin Ward 2018

class Unit(object):
    

    def __init__(self, v):
        self.val = v


    def prev_layer(self, prev):
        self.prev = prev


    def next_layer(self, next):
        self.next = next


i_layer = [Unit(0), Unit(0)]
h_layer = [Unit(0), Unit(0)]
o_layer = [Unit(0)]

for i, h in zip(i_layer, h_layer):
    i.next_layer(h_layer)
    h.prev_layer(i_layer)
    h.next_layer(o_layer)


