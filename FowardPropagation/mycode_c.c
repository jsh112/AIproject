#include <stdio.h>
#include <stdlib.h>
#include <math.h>

/**
 * @brief Structure to represent a neural network layer.
 *
 * @param currentSize Current number of nodes in the layer.
 * @param nextSize Number of nodes in the next layer.
 * @param weights Weights matrix [next_size x current_size].
 * @param input Input values stored (from previous layer or initial input).
 */

typedef struct Layer
{
    int currentSize;
    int nextSize;
    float **weights;
    float *input;
} Layer;

/**
 * @brief Reads layer information from the specified file.
 *
 * @param file Pointer to the file containing layer information.
 * @param nLayer Pointer to an integer where the number of layers will be stored.
 * @param NodeInfo Pointer to an array where the number of nodes in each layer will be stored.
 * @param Layers Pointer to an array of Layer structures that will be allocated.
 * @return int Returns 0 on success.
 */
int ReadLayerInfo(FILE *file, int *nLayer, int **NodeInfo, Layer **Layers)
{
    // Reading the number of layers
    fscanf(file, "%d", nLayer);
    *NodeInfo = (int *)malloc(sizeof(int) * (*nLayer));

    // Reading the number of nodes in each layer
    for (int i = 0; i < *nLayer; i++)
    {
        fscanf(file, "%d", &(*NodeInfo)[i]);
    }

    // Dynamic allocation of layer array
    *Layers = (Layer *)malloc((*nLayer) * sizeof(Layer));
    return 0;
}

/**
 * @brief Allocates memory for weights and initializes layer sizes.
 *
 * @param file Pointer to the file containing weights.
 * @param nLayer Total number of layers.
 * @param NodeInfo Array containing the number of nodes in each layer.
 * @param layers Pointer to the array of Layer structures.
 */
void AllocateWeights(FILE *file, int nLayer, int *NodeInfo, Layer *layers)
{
    for (int i = 0; i < nLayer - 1; i++)
    {
        layers[i].currentSize = NodeInfo[i];
        layers[i].nextSize = NodeInfo[i + 1];

        int sRow = layers[i].nextSize;
        int sCol = layers[i].currentSize;

        layers[i].weights = (float **)malloc(sRow * sizeof(float *));
        for (int j = 0; j < sRow; j++)
        {
            layers[i].weights[j] = (float *)malloc(sCol * sizeof(float));
            // Read the weights
            for (int k = 0; k < sCol; k++)
            {
                fscanf(file, "%f", &layers[i].weights[j][k]);
            }
        }

        layers[i + 1].input = (float *)malloc(sRow * sizeof(float)); // Allocate memory for the input of the next layer.
    }
    int OutPutLayerIndex = nLayer - 1;
    layers[OutPutLayerIndex].currentSize = NodeInfo[OutPutLayerIndex];
    layers[OutPutLayerIndex].nextSize = 0;
    layers[OutPutLayerIndex].weights = NULL;
}

/**
 * @brief Reads input values into the first layer's input and target values for the output layer.
 *
 * @param file   Pointer to the file containing input values.
 * @param sInput integer to store the number of input layer's nodes
 * @param layers Pointer to the array of Layer structures.
 */
void ReadInputValues(FILE *file, int sInput, Layer *layers)
{
    layers[0].input = (float *)malloc(sInput * sizeof(float));
    for (int i = 0; i < sInput; i++)
    {
        fscanf(file, "%f", &layers[0].input[i]);
    }
}

/**
 * @brief Reads input values into the first layer's input and target values for the output layer.
 *
 * @param file Pointer to the file containing input values.
 * @param NodeInfo Array containing the number of nodes in each layer.
 * @param nLayer Total number of layers.
 */
void ReadTargetValues(FILE *file, int NodeOutput, float *target)
{
    for (int idx = 0; idx < NodeOutput; idx++)
    {
        fscanf(file, "%f", &target[idx]);
    }
}

/**
 * @brief Rounds a float value to four decimal places.
 *
 * @param value The float value to be rounded.
 * @return float The rounded value.
 */
float roundToDecimals(float value)
{
    return roundf(value * 10000) / 10000;
}

/**
 * @brief Computes the sigmoid activation function.
 *
 * @param sum The input value for the sigmoid function.
 * @return float The result of the sigmoid function.
 */
float sigmoid(float sum)
{
    return roundToDecimals(1.0 / (1.0 + exp(-sum)));
}

/**
 * @brief Performs forward propagation through the network.
 *
 * @param nLayer Total number of layers.
 * @param layers Pointer to the array of Layer structures.
 */
void ForwardPropagation(int nLayer, Layer *layers)
{
    for (int i = 0; i < nLayer - 1; i++)
    {
        for (int j = 0; j < layers[i].nextSize; j++)
        {
            float sum = 0.0;
            for (int k = 0; k < layers[i].currentSize; k++)
            {
                sum += layers[i].input[k] * layers[i].weights[j][k];
            }
            layers[i + 1].input[j] = sigmoid(sum); // Next layer's input is this layer's output
            printf("%.4f ", layers[i + 1].input[j]);
        }
        printf("\n");
    }
}

void BackPropagation(int nLayer, Layer *layers)
{
}

/**
 * @brief Prints the values for the output layer.
 *
 * @param nLayer Total number of layers.
 * @param layers Pointer to the array of Layer structures.
 */
void PrintOutputLayer(int nLayer, Layer *layers)
{
    int OutputLayerIndex = nLayer - 1;
    printf("Values for the Output layer (Layer %d):\n", OutputLayerIndex + 1);

    for (int i = 0; i < layers[OutputLayerIndex - 1].nextSize; i++)
    {
        printf("Output %d: %.4f\n", i + 1, layers[OutputLayerIndex].input[i]);
    }
}

/**
 * @brief Frees allocated memory for layers and node information.
 *
 * @param nLayer Total number of layers.
 * @param layers Pointer to the array of Layer structures.
 * @param NodeInfo Pointer to the array of node information.
 */
void FreeMemory(int nLayer, Layer *layers, int *NodeInfo, float *target)
{
    for (int i = 0; i < nLayer; i++)
    {
        free(layers[i].input);
        if (i < nLayer - 1)
        {
            for (int j = 0; j < layers[i].nextSize; j++)
            {
                free(layers[i].weights[j]);
            }
            free(layers[i].weights);
        }
    }
    free(target);
    free(NodeInfo);
    free(layers);
}

/**
 * @brief Main function to execute the neural network program.
 *
 * @return int Returns 0 on successful execution.
 */
int main()
{
    int nLayer, *NodeInfo;
    Layer *Layers;
    float *targets;
    char filename[100];
    scanf("%s", filename);
    FILE *file = fopen(filename, "r");

    if (file == NULL)
    {
        perror("Error opening file");
        return 1;
    }

    if (ReadLayerInfo(file, &nLayer, &NodeInfo, &Layers) != 0)
    {
        fclose(file);
        return 1;
    }

    AllocateWeights(file, nLayer, NodeInfo, Layers);
    int sInput = NodeInfo[0];
    ReadInputValues(file, sInput, Layers);
    // ----------------------------------------------
    int Nodeoutput = NodeInfo[nLayer - 1];
    targets = (float *)malloc(sizeof(float) * Nodeoutput);
    ReadTargetValues(file, Nodeoutput, targets);
    // ----------------------------------------------

    fclose(file);

    // Perform forward propagation
    ForwardPropagation(nLayer, Layers);

    // Print the final layer's input values (output of the network)
    PrintOutputLayer(nLayer, Layers);

    // Free allocated memory
    FreeMemory(nLayer, Layers, NodeInfo, targets);

    return 0;
}
