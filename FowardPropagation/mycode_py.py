import numpy as np
import json
import sympy as sp


class NeuralNetwork:
    def __init__(self, learning_rate=0.001):
        with open('C:/AI_project/FowardPropagation/info.json', 'r') as file:
            json_data = json.load(file)
        nLayer = json_data['n']
        NodeInfo = json_data['numbers']
        inputs = json_data['input']
        targets = json_data['target']

        self.size = nLayer
        self.NodeInfo = NodeInfo
        self.input = np.array(inputs).T
        self.learning_rate = learning_rate
        self.target = targets
        self.weight = [np.random.randn(
            self.NodeInfo[i+1], self.NodeInfo[i]) for i in range(self.size-1)]

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def ForwardPropagation(self):
        for i in range(self.size-1):
            self.input = self.sigmoid(np.dot(self.weight[i], self.input))
            # append input

    def BackPropagation(self):
        """ 
        for i in range(self.size-2, -1, -1):
            for j in range(self.weight[i].shape[0]):
                for k in range(self.weight[i].shape[1]):
                    self.weight[i][j][k] -= -2 * (self.target[i] - input) 
        """
        """
            
        """


nn = NeuralNetwork(0.01)
nn.ForwardPropagation()

np.set_printoptions(precision=4, suppress=True)
print(f"{nn.input}")

# x = sp.symbols('x')
# print(sp.diff(1/(1+sp.exp(-x))))  # f'(x) = f(x) * (1 - f(x))
