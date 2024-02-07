#include "ann.h"

#include <stdio.h>
#include <stdlib.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>

#include "data.h"
#include "parameters.h"
#include "utils.h"

float maxMat(float **mat, int r, int c) {
    float max = 0.0f;
    for (int i = 0; i < r; ++i) {
        for (int j = 0; j < c; ++j) {
            max = (mat[i][j] > max) ? mat[i][j] : max;
        }
    }
    return max;
}

void initParams(Params *p) {
    p->W1 = initRandom(HL_SIZE, N);
    p->b1 = initRandom(HL_SIZE, 1);
    p->W2 = initRandom(HL_SIZE, HL_SIZE);
    p->b2 = initRandom(HL_SIZE, 1);
}

void forwardProp(Params *p, float **X, int m) {
    if (p->Z1 != NULL) {
        freeMatf(p->Z1, HL_SIZE);
        freeMatf(p->A1, HL_SIZE);
        freeMatf(p->Z2, HL_SIZE);
        freeMatf(p->A2, HL_SIZE);
    }

    float **W1dotX = dot(p->W1, HL_SIZE, N, X, N, m);
    p->Z1 = addBias(W1dotX, HL_SIZE, m, p->b1);

    p->A1 = relu(p->Z1, HL_SIZE, m);

    float **W2dotA1 = dot(p->W2, HL_SIZE, HL_SIZE, p->A1, HL_SIZE, m);
    p->Z2 = addBias(W2dotA1, HL_SIZE, m, p->b2);

    p->A2 = softmax(p->Z2, HL_SIZE, m);

    freeMatf(W1dotX, HL_SIZE);
    freeMatf(W2dotA1, HL_SIZE);
}

void backProp(Params *p, float **X, float **Y) {  // Y HL_SIZE x 1
    if (p->dW2 != NULL) {
        freeMatf(p->dW2, HL_SIZE);
        freeMatf(p->dW1, HL_SIZE);
        freeMatf(p->db1, HL_SIZE);
        freeMatf(p->db2, HL_SIZE);
    }
    float **dZ2 = sub(p->A2, Y, HL_SIZE, M_TRAIN);

    float **A1t = transposef(p->A1, HL_SIZE, M_TRAIN);
    float **dZ2dotA1t = dot(dZ2, HL_SIZE, M_TRAIN, A1t, M_TRAIN, HL_SIZE);
    p->dW2 = scalarMult(dZ2dotA1t, HL_SIZE, HL_SIZE, 1.0f / M_TRAIN);

    freeMatf(A1t, M_TRAIN);
    freeMatf(dZ2dotA1t, HL_SIZE);

    float **dZ2rs = sumRows(dZ2, HL_SIZE, M_TRAIN);
    p->db2 = scalarMult(dZ2rs, HL_SIZE, M_TRAIN, 1.0f / M_TRAIN);
    freeMatf(dZ2rs, HL_SIZE);

    float **W2t = transposef(p->W2, HL_SIZE, HL_SIZE);
    float **W2tdotdZ2 = dot(W2t, HL_SIZE, HL_SIZE, dZ2, HL_SIZE, M_TRAIN);
    float **dZdrelu = drelu(p->Z1, HL_SIZE, M_TRAIN);
    float **dZ1 = mult(W2tdotdZ2, dZdrelu, HL_SIZE, M_TRAIN);
    freeMatf(W2t, HL_SIZE);
    freeMatf(W2tdotdZ2, HL_SIZE);
    freeMatf(dZdrelu, HL_SIZE);

    float **Xt = transposef(X, N, M_TRAIN);
    float **dZ1dotXt = dot(dZ1, HL_SIZE, M_TRAIN, Xt, M_TRAIN, N);
    p->dW1 = scalarMult(dZ1dotXt, HL_SIZE, N, 1.0f / M_TRAIN);
    freeMatf(Xt, M_TRAIN);
    freeMatf(dZ1dotXt, HL_SIZE);

    float **dZ1rs = sumRows(dZ1, HL_SIZE, M_TRAIN);
    p->db1 = scalarMult(dZ1rs, HL_SIZE, M_TRAIN, 1.0f / M_TRAIN);
    freeMatf(dZ1rs, HL_SIZE);

    freeMatf(dZ2, HL_SIZE);
    freeMatf(dZ1, HL_SIZE);
}

void updateParams(Params *p, float alpha) {
    float **adW1 = scalarMult(p->dW1, HL_SIZE, N, alpha);
    float **W1subadW1 = sub(p->W1, adW1, HL_SIZE, N);
    freeMatf(p->W1, HL_SIZE);
    p->W1 = W1subadW1;

    float **adb1 = scalarMult(p->db1, HL_SIZE, 1, alpha);
    float **ab1 = sub(p->b1, adb1, HL_SIZE, 1);
    freeMatf(p->b1, HL_SIZE);
    freeMatf(adb1, HL_SIZE);
    p->b1 = ab1;

    float **adW2 = scalarMult(p->dW2, HL_SIZE, HL_SIZE, alpha);
    float **W2subadW2 = sub(p->W2, adW2, HL_SIZE, HL_SIZE);
    freeMatf(p->W2, HL_SIZE);
    p->W2 = W2subadW2;

    float **adb2 = scalarMult(p->db2, HL_SIZE, 1, alpha);
    float **ab2 = sub(p->b2, adb2, HL_SIZE, 1);
    freeMatf(p->b2, HL_SIZE);
    freeMatf(adb2, HL_SIZE);
    p->b2 = ab2;
}

float getAccuracy(int *pred, int *Y, int r) {
    int sum = 0;
    for (int i = 0; i < r; ++i) {
        sum += pred[i] == Y[i];
    }
    return sum * 1.0f / r;
}

void gradientDescent(Params *p, float **X, float **YoneHot, int *Y, float alpha,
                     int iterations) {
    printf("Training Model...\n");
    initParams(p);
    for (int i = 0; i < iterations; ++i) {
        forwardProp(p, X, M_TRAIN);
        backProp(p, X, YoneHot);
        updateParams(p, alpha);
        if (i % 10 == 0) {
            printf("Iteration %d: ", i);
            int *pred = argmax(p->A2, HL_SIZE, M_TRAIN);
            float acc = getAccuracy(pred, Y, M_TRAIN);
            free(pred);
            printf("Accuracy: %f\n", acc);
        }
    }
    printf("\n");
}

void evaluate(Params *p, float **X, float **YoneHot, int *Y) {
    printf("Evaluating Model...\n");
    forwardProp(p, X, M_TEST);
    int *pred = argmax(p->A2, HL_SIZE, M_TEST);
    float acc = getAccuracy(pred, Y, M_TEST);
    free(pred);
    printf("Accuracy: %f\n", acc);
    printf("\n");
}

float *predict(Params *p, float **x) {
    float *res = (float *)calloc(2, sizeof(float));
    forwardProp(p, x, 1);
    int *pred = argmax(p->A2, HL_SIZE, 1);
    res[0] = pred[0];
    res[1] = p->A2[pred[0]][0];
    free(pred);
    return res;
}

void writeParams(Params *p, const char *filename) {
    // printf("Saving model...\n");
    // printf("Loading directory... ");
    struct stat st = {0};
    if (stat("/models", &st) == -1) {
        mkdir("models", 0777);
    }

    char filepath[100];
    sprintf(filepath, "./models/%s", filename);
    FILE *file = fopen(filepath, "w");
    if (file == NULL) {
        perror("Error opening file");
        return;
    }
    printf("Complete!\n");
    printf("Writing... ");
    for (int i = 0; i < HL_SIZE; ++i) {
        for (int j = 0; j < N; ++j) {
            fprintf(file, "%f ", p->W1[i][j]);
        }
        fprintf(file, "\n");
    }
    for (int i = 0; i < HL_SIZE; ++i) {
        fprintf(file, "%f ", p->b1[i][0]);
    }
    fprintf(file, "\n");
    for (int i = 0; i < HL_SIZE; ++i) {
        for (int j = 0; j < HL_SIZE; ++j) {
            fprintf(file, "%f ", p->W2[i][j]);
        }
        fprintf(file, "\n");
    }
    for (int i = 0; i < HL_SIZE; ++i) {
        fprintf(file, "%f ", p->b2[i][0]);
    }
    fprintf(file, "\n");
    printf("Complete!\n");
    fclose(file);
}

void readParams(Params *p, const char *filename) {
    printf("Loading file... ");
    char filepath[100];
    sprintf(filepath, "./models/%s", filename);
    FILE *file = fopen(filepath, "r");
    if (file == NULL) {
        perror("Error opening file");
        return;
    }
    printf("Complete!\n");
    initParams(p);
    for (int i = 0; i < HL_SIZE; ++i) {
        for (int j = 0; j < N; ++j) {
            if (fscanf(file, "%f", &p->W1[i][j]) != 1) {
                fprintf(stderr, "Error reading float W1");
                fclose(file);
                return;
            }

            char space;
            if (fscanf(file, "%c", &space) != 1 || space != ' ') {
                fprintf(stderr, "Error reading space W1");
                fclose(file);
                return;
            }
        }
        char newline;
        if (fscanf(file, "%c", &newline) != 1 || newline != '\n') {
            fprintf(stderr, "Error reading newline W1");
            fclose(file);
            return;
        }
    }

    for (int i = 0; i < HL_SIZE; ++i) {
        if (fscanf(file, "%f", &p->b1[i][0]) != 1) {
            fprintf(stderr, "Error reading float b1");
            fclose(file);
            return;
        }

        char space;
        if (fscanf(file, "%c", &space) != 1 || space != ' ') {
            fprintf(stderr, "Error reading space b1");
            fclose(file);
            return;
        }
    }
    char newline;
    if (fscanf(file, "%c", &newline) != 1 || newline != '\n') {
        fprintf(stderr, "Error reading newline b1");
        fclose(file);
        return;
    }

    for (int i = 0; i < HL_SIZE; ++i) {
        for (int j = 0; j < HL_SIZE; ++j) {
            if (fscanf(file, "%f", &p->W2[i][j]) != 1) {
                fprintf(stderr, "Error reading float W2");
                fclose(file);
                return;
            }

            char space;
            if (fscanf(file, "%c", &space) != 1 || space != ' ') {
                fprintf(stderr, "Error reading space W2");
                fclose(file);
                return;
            }
        }
        char newline;
        if (fscanf(file, "%c", &newline) != 1 || newline != '\n') {
            fprintf(stderr, "Error reading newline W2");
            fclose(file);
            return;
        }
    }
    for (int i = 0; i < HL_SIZE; ++i) {
        if (fscanf(file, "%f", &p->b2[i][0]) != 1) {
            fprintf(stderr, "Error reading float b2");
            fclose(file);
            return;
        }

        char space;
        if (fscanf(file, "%c", &space) != 1 || space != ' ') {
            fprintf(stderr, "Error reading space b2");
            fclose(file);
            return;
        }
    }
    if (fscanf(file, "%c", &newline) != 1 || newline != '\n') {
        fprintf(stderr, "Error reading newline b2");
        fclose(file);
        return;
    }
    fclose(file);
}
