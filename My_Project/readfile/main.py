from neural_network import NeuralNetwork
import numpy as np

'''
    The file consists 
    0. numbers of layers
    1. numbers of nodes for each layers
    2 ~ [-3]. the (nextSize * currentSize) matrix of each layes
    -2. numbers of input nodes
    -1. numbers of target nodes
'''


def main():
    filename = input("Enter filename: ")
    with open(filename, 'r') as file:
        lines = file.readlines()

    nn = NeuralNetwork(lines ,learning_rate=0.05)
    
    print(f"Initial Input = {nn.layers[0].input}")
    print(f"Target = {nn.target}")

    print(f'First FP is : {nn.forward_propagation()}')
    nn.back_propagation()

    nn.train(epochs=500)
    
    nn.print_final_state()


if __name__ == '__main__':
    main()
