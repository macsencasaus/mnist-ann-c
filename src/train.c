#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#include "ann.h"
#include "data.h"
#include "parameters.h"

int main(int argc, char *argv[]) {
    srand(time(NULL));

    Data tr = {NULL};
    Data te = {NULL};
    getData(&tr, &te);

    Params p = {NULL};
    gradientDescent(&p, tr.X, tr.YoneHot, tr.Y, LEARNING_RATE, ITERATIONS);
    evaluate(&p, te.X, te.YoneHot, te.Y);

    const char *filename;
    if (argc > 1 && argv[1][0] != '\0') {
        filename = argv[1];
    } else {
        filename = "model.txt";
    }
    writeParams(&p, filename);
    printf(
        "Run ./predict <index> (0 <= index <= %d) in order to apply the "
        "model to the testing data!\n",
        M_PRED - 1);
    return 0;
}
