#include "ann.h"
#include "data.h"
#include "stdio.h"
#include "stdlib.h"
#include "utils.h"

int main(int argc, char *argv[]) {
    const char *filename = "model.txt";
    Params p = {NULL};
    readParams(&p, filename);

    int idx = 0;
    if (argc > 1) {
        idx = atoi(argv[1]);
    }

    Data t = {NULL};
    getTest(&t);

    float **x = transposeArrf(t.X[idx], N);
    float *pred = predict(&p, x);
    printf("Image: \n");
    printImage(t.X[idx]);
    printf("Prediction: %d\nConfidence: %f\n", (int)pred[0], pred[1]);
    free(pred);
    freeMatf(x, 1);
}
