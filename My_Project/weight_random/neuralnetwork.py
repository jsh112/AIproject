import numpy as np


class Layer:
    def __init__(self):
        self.currentSize = cSize
        self.nextSize = nSize
        self.weights = np.random.rand(nSize, cSize) if nSize > 0 else None
        self.input = np.zeros(cSize)
        self.z = None
        self.activation = None
        self.delta = None


class Neural_Network:
    def __init__(self, lines, learnig_rate=0.01):
        self.data = lines
        self.node_info = [int(x) for x in self.data[1].split()]
        self.layers = [
            Layer(
                self.node_info[i],
                self.node_info[i + 1] if i < len(self.node_info) - 1 else 0
            )
            for i in range(len(self.node_info))
        ]
        self.learning_rate = learning_rate
        
    def allocate_weights(self):
        pass
