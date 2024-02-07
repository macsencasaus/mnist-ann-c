#ifndef UTILS_H
#define UTILS_H

float **makeMatf(int r, int c);

int **makeMat(int r, int c);

void freeMatf(float **mat, int r);

void freeMat(int **mat, int r);

float **transposef(float **mat, int r, int c);

float **transposeArrf(float *mat, int n);

float **relu(float **mat, int r, int c);

float **drelu(float **mat, int r, int c);

float **dot(float **mat1, int r1, int c1, float **mat2, int r2, int c2);

float **add(float **mat1, float **mat2, int r, int c);

float **sub(float **mat1, float **mat2, int r, int c);

float **mult(float **mat1, float **mat2, int r, int c);

float **initRandom(int r, int c);

float **sumRows(float **mat, int r, int c);

float **softmax(float **mat, int r, int c);

float **scalarMult(float **mat, int r, int c, float a);

float **addBias(float **mat, int r, int c, float **b);

int *argmax(float **mat, int r, int c);

void printImage(float *mat);

#endif
