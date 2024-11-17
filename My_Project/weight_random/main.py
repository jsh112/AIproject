import numpy as np
import matplotlib.pyplot as plt
from neuralnetwork import Neural_Network


def main():
    filename = input("Enter filename: ")
    with open(filename, 'r') as file:
        lines = file.readlines()

    nn = Neural_Network(lines, learning_rate=0.01)


if __name__ == '__main__':
    main()
