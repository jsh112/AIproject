import numpy as np


class Layer:
    def __init__(self, currentSize, nextSize):
        self.currentSize = currentSize
        self.nextSize = nextSize
        self.weights = np.zeros(
            (self.nextSize, self.currentSize)) if nextSize > 0 else None
        self.input = np.zeros(currentSize)
        self.delta = np.zeros(currentSize) if nextSize > 0 else None


def ReadLayerInfo(lines):
    nLayer = int(lines[0].strip())
    NodeInfo = [int(x) for x in lines[1].split()]
    Layers = []

    for i in range(nLayer):
        currentSize = NodeInfo[i]
        nextSize = NodeInfo[i + 1] if i < nLayer - 1 else 0
        layer = Layer(currentSize, nextSize)
        layer.input = np.zeros(currentSize)
        Layers.append(layer)
    return nLayer, NodeInfo, Layers


def allocate_weights(lines, Layers):
    index = 2
    weights = [float(x) for x in ' '.join(lines[index:]).split()]
    weight_index = 0

    for i in range(len(Layers) - 1):
        sRow = Layers[i].nextSize
        sCol = Layers[i].currentSize
        expected_weights = sRow * sCol

        layer_weights = np.array(
            weights[weight_index:weight_index + expected_weights]).reshape(sRow, sCol)
        Layers[i].weights = layer_weights
        weight_index += expected_weights

        print(f"Weights for Layer {i} (shape {Layers[i].weights.shape}):")
        print(Layers[i].weights)


def ReadValues(lines, NodeInfo, layers):
    input_values = [float(x) for x in lines[-2].strip().split()]
    target_values = [float(x) for x in lines[-1].strip().split()]

    # Check
    if len(input_values) != NodeInfo[0]:
        raise ValueError(
            f'Expected {NodeInfo[0]} input values, but got {len(input_values)}')
    if len(target_values) != NodeInfo[-1]:
        raise ValueError(
            f'Expected {NodeInfo[-1]} input values, but got {len(target_values)}')

    layers[0].input = np.array(input_values)
    return np.array(target_values)


def roundToDecimals(value):
    return round(value * 100000000) / 100000000


def sigmoid(sum):
    return roundToDecimals(1.0 / (1.0 + np.exp(-sum)))


def ForwardPropagation(nLayer, Layers):
    for i in range(nLayer - 1):
        for j in range(Layers[i].nextSize):
            sum = 0.0
            for k in range(Layers[i].currentSize):
                sum += (Layers[i].input[k] * Layers[i].weights[j][k])
            Layers[i+1].input[j] = sigmoid(sum)
            print(f'{Layers[i + 1].input[j]:.8f} ', end='')
        print()


def main():
    filename = input("Enter filename : ")
    with open(filename, 'r') as file:
        # Like FILE *file
        lines = file.readlines()

    nLayer, NodeInfo, Layers = ReadLayerInfo(lines)

    allocate_weights(lines, Layers)

    target = ReadValues(lines, NodeInfo, Layers)
    print(f'input = {Layers[0].input}')
    print(f'target = {target}')

    ForwardPropagation(nLayer, Layers)


if __name__ == '__main__':
    main()
