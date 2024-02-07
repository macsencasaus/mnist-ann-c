#ifndef DATA_H
#define DATA_H

#define N 784

typedef struct {
    float **X;
    float **YoneHot;
    int *Y;
} Data;

void getData(Data *tr, Data *te);

void getTest(Data *t);

#endif
