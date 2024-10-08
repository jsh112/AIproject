#include <stdio.h>
#include <stdlib.h>
#include <math.h>

typedef struct Layer
{
    int currentSize; // currentSize is number of nodes in current layer
    int nextSize;    // nextSize is number of nodes in next layer
    float **weights; // weights matrix [next_size x current_size]
    float *input;    // store the input values (from previous layer or initial input)
} Layer;

// Read nLayer, NodeInfo array, Layers array
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

// Allocate weights implementation
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

// Read input values into the first layer's input
void ReadInputValues(FILE *file, int *NodeInfo, Layer *layers)
{
    int sInput = NodeInfo[0];
    layers[0].input = (float *)malloc(sInput * sizeof(float));
    for (int i = 0; i < sInput; i++)
    {
        fscanf(file, "%f", &layers[0].input[i]);
    }
}

float roundToDecimals(float value)
{
    return roundf(value * 10000) / 10000;
}

float sigmoid(float sum)
{
    return roundToDecimals(1.0 / (1.0 + exp(-sum)));
}

// Forward propagation
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
            layers[i + 1].input[j] = roundToDecimals(sum); // Next layer's input is this layer's output
            printf("%.4f ", layers[i + 1].input[j]);
        }
        printf("\n");
    }
}

// Print the final layer's input
void PrintOutputLayer(int nLayer, Layer *layers)
{
    int OutputLayerIndex = nLayer - 1;
    printf("Values for the Output layer (Layer %d):\n", OutputLayerIndex + 1);

    for (int i = 0; i < layers[OutputLayerIndex - 1].nextSize; i++)
    {
        printf("Output %d: %.4f\n", i + 1, layers[OutputLayerIndex].input[i]);
    }
}

void FreeMemory(int nLayer, Layer *layers, int *NodeInfo)
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
    free(NodeInfo);
    free(layers);
}

int main()
{
    int nLayer, *NodeInfo;
    Layer *Layers;
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
    ReadInputValues(file, NodeInfo, Layers);
    fclose(file);

    // Perform forward propagation
    ForwardPropagation(nLayer, Layers);

    // Print the final layer's input values (output of the network)
    PrintOutputLayer(nLayer, Layers);

    // Free allocated memory
    FreeMemory(nLayer, Layers, NodeInfo);

    return 0;
}
