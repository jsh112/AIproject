import numpy as np

class Layer:
    def __init__(self, cSize, nSize):
        self.currentSize = cSize
        self.nextSize = nSize
        self.weights = np.random.rand(nSize, cSize) if nSize > 0 else None
        self.input = np.zeros(cSize)
        self.z = None
        self.activation = None
        self.delta = None

class NeuralNetwork:
    def __init__(self, lines, learning_rate=0.01):
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

        # Read input value and target values
        self.read_values()
        # Allocate initial weights
        self.allocate_weights()

    # Read input value and target values
    def read_values(self):
        _input = np.array([float(x) for x in self.data[-2].strip().split()])
        _target = np.array([float(x) for x in self.data[-1].strip().split()])

        if len(_input) != self.node_info[0] or len(_target) != self.node_info[-1]:
            raise ValueError("Mismatch in expected input or target values.")

        self.layers[0].input = _input
        self.target = _target

    # Allocate initial weights
    def allocate_weights(self):
        weights = [float(x) for x in ' '.join(self.data[2:-2]).split()]
        weights_index = 0

        for i in range(len(self.layers) - 1):
            sRow, sCol = self.layers[i].nextSize, self.layers[i].currentSize
            matrix_size = sRow * sCol
            layers_weights = np.array(
                weights[weights_index:weights_index + matrix_size]
            ).reshape(sRow, sCol)

            self.layers[i].weights = layers_weights
            weights_index += matrix_size

            print(f'Weights of layer {i} (shape {layers_weights.shape}):\n{self.layers[i].weights}\n')

    # Forward propagation
    def forward_propagation(self):
        for i in range(len(self.layers) - 1):
            current_layer = self.layers[i]
            next_layer = self.layers[i + 1]

            next_layer.z = np.dot(current_layer.weights, current_layer.input)
            next_layer.activation = self.LReLU(next_layer.z)
            next_layer.input = next_layer.activation

        return self.layers[-1].activation

    # Backpropagation
    def back_propagation(self):
        # Output layer delta
        output_layer = self.layers[-1]
        error = output_layer.activation - self.target
        derivative = self.LReLU_derivative(output_layer.z)
        output_layer.delta = error * derivative

        # Hidden layers delta (excluding input layer)
        for i in reversed(range(1, len(self.layers) - 1)):
            current_layer = self.layers[i]
            next_layer = self.layers[i + 1]
            if current_layer.weights is not None:
                current_layer.delta = np.dot(
                    current_layer.weights.T, next_layer.delta) * self.LReLU_derivative(current_layer.z)

        # Weight updates
        for i in range(len(self.layers) - 1):
            current_layer = self.layers[i]
            next_layer = self.layers[i + 1]

            # Previous layer activation
            if i == 0:
                a_prev = current_layer.input
            else:
                a_prev = self.layers[i].activation

            # Gradient calculation
            gradient = np.outer(next_layer.delta, a_prev)

            # Weight update
            if current_layer.weights.shape == gradient.shape:
                current_layer.weights -= self.learning_rate * gradient
            else:
                print(f'Error: Shape mismatch when updating weights for layer {i}.')
                print(f'Weights shape: {current_layer.weights.shape}, Gradient shape: {gradient.shape}\n')

    # Calculate loss (Mean Squared Error)
    def calculate_loss(self):
        error = self.layers[-1].activation - self.target
        loss = np.mean(error ** 2)
        return loss

    # Leaky ReLU function
    def LReLU(self, x):
        return np.where(x > 0, x, 0.01 * x)

    # The derivative of Leaky ReLU
    def LReLU_derivative(self, x):
        return np.where(x > 0, 1.0, 0.01)

    # Training method
    def train(self, epochs=10):
        for epoch in range(1, epochs + 1):
            # Forward propagation
            output = self.forward_propagation()

            # Backpropagation
            self.back_propagation()

            # Calculate loss
            loss = self.calculate_loss()

            # Print output and loss
            print(f'Epoch {epoch}/{epochs}, Output: {output}, Loss: {loss}\n')

            # Print updated weights after every epoch
            for i in range(len(self.layers) - 1):
                print(f'Updated Weights of layer {i}:\n{self.layers[i].weights}\n')

    # Optional: Function to print final weights and output
    def print_final_state(self):
        print("Final Weights:")
        for i in range(len(self.layers) - 1):
            print(f'Layer {i} weights (shape {self.layers[i].weights.shape}):\n{self.layers[i].weights}\n')
        print(f'Final Output: {self.layers[-1].activation}')
