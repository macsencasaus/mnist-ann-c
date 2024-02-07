#include "utils.h"

#include <stdio.h>
#include <stdlib.h>

#include "math.h"

float **makeMatf(int r, int c) {
    float **res = (float **)calloc(r, sizeof(float *));
    for (int i = 0; i < r; ++i) {
        res[i] = (float *)calloc(c, sizeof(float));
    }
    return res;
}

int **makeMat(int r, int c) {
    int **res = (int **)calloc(r, sizeof(int *));
    for (int i = 0; i < r; ++i) {
        res[i] = (int *)calloc(c, sizeof(int));
    }
    return res;
}

void freeMatf(float **mat, int r) {
    for (int i = 0; i < r; ++i) {
        free(mat[i]);
    }
    free(mat);
}

void freeMat(int **mat, int r) {
    for (int i = 0; i < r; ++i) {
        free(mat[i]);
    }
    free(mat);
}

float **transposef(float **mat, int r, int c) {
    float **res = makeMatf(c, r);
    for (int i = 0; i < r; ++i) {
        for (int j = 0; j < c; ++j) {
            res[j][i] = mat[i][j];
        }
    }
    return res;
}

float **transposeArrf(float *mat, int n) {
    float **res = makeMatf(n, 1);
    for (int i = 0; i < n; ++i) {
        res[i][0] = mat[i];
    }
    return res;
}

float **relu(float **mat, int r, int c) {
    float **res = makeMatf(r, c);
    for (int i = 0; i < r; ++i) {
        for (int j = 0; j < c; ++j) {
            float x = mat[i][j];
            res[i][j] = (x > 0) ? x : 0.0f;
        }
    }
    return res;
}

float **drelu(float **mat, int r, int c) {
    float **res = makeMatf(r, c);
    for (int i = 0; i < r; ++i) {
        for (int j = 0; j < c; ++j) {
            float x = mat[i][j];
            res[i][j] = (x > 0) ? 1.0f : 0.0f;
        }
    }
    return res;
}

float **dot(float **mat1, int r1, int c1, float **mat2, int r2, int c2) {
    if (c1 != r2) return NULL;
    float **res = makeMatf(r1, c2);
    for (int i = 0; i < r1; ++i) {
        for (int j = 0; j < c2; ++j) {
            res[i][j] = 0.0f;
            for (int h = 0; h < c1; ++h) {
                res[i][j] += mat1[i][h] * mat2[h][j];
            }
        }
    }
    return res;
}

float **add(float **mat1, float **mat2, int r, int c) {
    float **res = makeMatf(r, c);
    for (int i = 0; i < r; ++i) {
        for (int j = 0; j < c; ++j) {
            res[i][j] = mat1[i][j] + mat2[i][j];
        }
    }
    return res;
}

float **sub(float **mat1, float **mat2, int r, int c) {
    float **res = makeMatf(r, c);
    for (int i = 0; i < r; ++i) {
        for (int j = 0; j < c; ++j) {
            res[i][j] = mat1[i][j] - mat2[i][j];
        }
    }
    return res;
}

float **mult(float **mat1, float **mat2, int r, int c) {
    float **res = makeMatf(r, c);
    for (int i = 0; i < r; ++i) {
        for (int j = 0; j < c; ++j) {
            res[i][j] = mat1[i][j] * mat2[i][j];
        }
    }
    return res;
}

float randf() { return rand() * 1.0f / RAND_MAX; }

float **initRandom(int r, int c) {
    float **res = makeMatf(r, c);
    for (int i = 0; i < r; ++i) {
        for (int j = 0; j < c; ++j) {
            res[i][j] = randf() - 0.5f;
        }
    }
    return res;
}

float **sumRows(float **mat, int r, int c) {
    float **res = makeMatf(r, 1);
    for (int i = 0; i < r; ++i) {
        float rowSum = 0;
        for (int j = 0; j < c; ++j) {
            rowSum += mat[i][j];
        }
        res[i][0] = rowSum;
    }
    return res;
}

float **softmax(float **mat, int r, int c) {
    float *rowSums = (float *)calloc(c, sizeof(float));
    for (int j = 0; j < c; ++j) {
        for (int i = 0; i < r; ++i) {
            rowSums[j] += expf(mat[i][j]);
        }
    }
    float **res = makeMatf(r, c);
    for (int i = 0; i < r; ++i) {
        for (int j = 0; j < c; ++j) {
            res[i][j] = expf(mat[i][j]) / rowSums[j];
        }
    }
    free(rowSums);
    return res;
}

float **scalarMult(float **mat, int r, int c, float a) {
    float **res = makeMatf(r, c);
    for (int i = 0; i < r; ++i) {
        for (int j = 0; j < c; ++j) {
            res[i][j] = mat[i][j] * a;
        }
    }
    return res;
}

float **addBias(float **mat, int r, int c, float **b) {
    float **res = makeMatf(r, c);
    for (int i = 0; i < r; ++i) {
        for (int j = 0; j < c; ++j) {
            res[i][j] = mat[i][j] + b[i][0];
        }
    }
    return res;
}

int *argmax(float **mat, int r, int c) {
    int *res = (int *)calloc(c, sizeof(int));
    for (int j = 0; j < c; j++) {
        float max = 0;
        int maxi = 0;
        for (int i = 0; i < r; i++) {
            if (mat[i][j] > max) {
                max = mat[i][j];
                maxi = i;
            }
        }
        res[j] = maxi;
    }
    return res;
}

void printImage(float *mat) {
    int h = 0;
    for (int i = 0; i < 28; ++i) {
        for (int j = 0; j < 28; ++j) {
            char x = (mat[h] > 0) ? (mat[h] > 0.5 ? '#' : '*') : ' ';
            h++;
            printf("%c ", x);
        }
        printf("\n");
    }
}
