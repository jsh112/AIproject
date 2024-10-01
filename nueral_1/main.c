#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>

typedef struct Layer
{
    int idx, size;
    float *nodes;
    float **weights;
    struct Layer *next_layer;
    struct Layer *prev_layer;
} Layer;

int idx = 1;

void createLayer(int size, int next_size)
{
}

int main()
{
    // input 1 / hidden layer 1 / output 1
    int n_layer = 3;
    int NodeInfo[] = {2, 3, 2};
    float input_data[] = {1.0, 0.5};

    // Make input layer
    Layer *input, *head;
    input = (Layer *)malloc(sizeof(Layer));

    // idx of input layer is 1, input layer has NodeInfo[idx - 1] (this case is 2) nodes
    input->idx = idx;
    input->size = NodeInfo[idx - 1];

    // Tentative state is NULL
    input->next_layer = NULL;
    input->prev_layer = NULL;
    idx++;

    return 0;
}
