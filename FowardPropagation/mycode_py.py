import numpy as np
import json


class NeuralNetwork:
    def __init__(self):
        with open('C:/AI_project/FowardPropagation/info.json', 'r') as file:
            json_data = json.load(file)
        nLayer = json_data['n']
        NodeInfo = json_data['numbers']
        inputs = json_data['input']
        self.size = nLayer
        self.NodeInfo = NodeInfo
        self.input = np.array(inputs).T
        self.weight = [np.random.randn(
            self.NodeInfo[i+1], self.NodeInfo[i]) for i in range(self.size-1)]
        self.ForwardPropagation()

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def ForwardPropagation(self):
        for i in range(self.size-1):
            self.input = self.sigmoid(np.dot(self.weight[i], self.input))

    def BackPropagation(self):


nn = NeuralNetwork()
np.set_printoptions(precision=4, suppress=True)
print(f"{nn.input}")
