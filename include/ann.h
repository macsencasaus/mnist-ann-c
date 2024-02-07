#ifndef ANN_H
#define ANN_H

#define HL_SIZE 10

typedef struct {
    // Params
    float **W1;  // HL_SIZE x N
    float **b1;  // HL_SIZE x 1
    float **W2;  // HL_SIZE x HL_SIZE
    float **b2;  // HL_SIZE x 1

    // Forward Prop
    float **Z1;  // HL_SIZE x M
    float **A1;  // HL_SIZE x M
    float **Z2;  // HL_SIZE x M
    float **A2;  // HL_SIZE x M

    // BackwardProp
    float **dW2;  // HL_SIZE x HL_SIZE
    float **db2;  // HL_SIZE x 1
    float **dW1;  // HL_SIZE x N
    float **db1;  // HL_SIZE x 1
} Params;

void initParams(Params *p);

void gradientDescent(Params *p, float **X, float **YoneHot, int *Y, float alpha,
                     int iterations);

void evaluate(Params *p, float **X, float **YoneHot, int *Y);

float *predict(Params *p, float **x);

void writeParams(Params *p, const char *filename);

void readParams(Params *p, const char *filename);

#endif
