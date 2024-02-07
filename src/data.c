#include "data.h"

#include <stdio.h>
#include <stdlib.h>

#include "parameters.h"
#include "utils.h"

void getData(Data* tr, Data* te) {
    tr->X = makeMatf(N, M_TRAIN);
    tr->YoneHot = makeMatf(10, M_TRAIN);
    tr->Y = (int*)calloc(M_TRAIN, sizeof(int));

    te->X = makeMatf(N, M_TEST);
    te->YoneHot = makeMatf(10, M_TEST);
    te->Y = (int*)calloc(M_TEST, sizeof(int));

    FILE* file;
    int** labels = makeMat(M_TRAIN + M_TEST, 1);
    int** data = makeMat(M_TRAIN + M_TEST, N);
    int i, j;

    file = fopen("./data/train/train.csv", "r");
    if (file == NULL) {
        perror("Error opening file");
        return;
    }

    for (i = 0; i < M_TRAIN + M_TEST; ++i) {
        for (j = 0; j < N + 1; ++j) {
            if (j == 0) {
                if (fscanf(file, "%d", &labels[i][j]) != 1) {
                    fprintf(stderr,
                            "Error decimal reading from file to label\n");
                    fclose(file);
                    return;
                }
            } else {
                if (fscanf(file, "%d", &data[i][j]) != 1) {
                    fprintf(stderr,
                            "Error decimal reading from file to data\n");
                    fclose(file);
                    return;
                }
            }

            if (j < N) {
                char comma;
                if (fscanf(file, "%c", &comma) != 1 || comma != ',') {
                    fprintf(stderr, "Error 2 reading from file\n");
                    fclose(file);
                    return;
                }
            }
        }

        char carriageReturn;
        if (fscanf(file, "%c", &carriageReturn) != 1 ||
            carriageReturn != '\r') {
            fprintf(stderr, "Error 3 reading from file, row: %d, got:\"%c\"\n",
                    i, carriageReturn);
            fclose(file);
            return;
        }

        char newline;
        if (fscanf(file, "%c", &newline) != 1 || newline != '\n') {
            fprintf(stderr, "Error 4 reading from file, row: %d, got:\"%c\"\n",
                    i, newline);
            fclose(file);
            return;
        }
    }

    fclose(file);

    for (i = 0; i < M_TRAIN + M_TEST; ++i) {
        for (j = 0; j < N; ++j) {
            // transposed & normalized
            if (i < M_TRAIN)
                tr->X[j][i] = data[i][j] / 255.0f;
            else
                te->X[j][i - M_TRAIN] = data[i][j] / 255.0f;
        }
    }

    for (i = 0; i < M_TRAIN + M_TEST; ++i) {
        int x = labels[i][0];
        if (i < M_TRAIN)
            tr->Y[i] = x;
        else
            te->Y[i - M_TRAIN] = x;
        for (int j = 0; j < 10; ++j) {
            // one hot
            if (i < M_TRAIN) {
                if (j == x) {
                    tr->YoneHot[j][i] = 1.0f;
                    continue;
                }
                tr->YoneHot[j][i] = 0.0f;
            } else {
                if (j == x) {
                    te->YoneHot[j][i - M_TRAIN] = 1.0f;
                    continue;
                }
                te->YoneHot[j][i - M_TRAIN] = 0.0f;
            }
        }
    }

    freeMat(data, M_TRAIN);
    freeMat(labels, M_TRAIN);
}

void getTest(Data* t) {
    t->X = makeMatf(M_PRED, N);

    FILE* file;
    int** data = makeMat(M_PRED, N);
    int i, j;

    file = fopen("./data/test/test.csv", "r");
    if (file == NULL) {
        perror("Error opening file");
        return;
    }

    for (i = 0; i < M_PRED; ++i) {
        for (j = 0; j < N; ++j) {
            if (fscanf(file, "%d", &data[i][j]) != 1) {
                fprintf(stderr, "Error 2 reading from file\n");
                fclose(file);
                return;
            }
            if (j < N - 1) {
                char comma;
                if (fscanf(file, "%c", &comma) != 1 || comma != ',') {
                    fprintf(stderr, "Error 2 reading from file\n");
                    fclose(file);
                    return;
                }
            }
        }

        char carriageReturn;
        if (fscanf(file, "%c", &carriageReturn) != 1 ||
            carriageReturn != '\r') {
            fprintf(stderr, "Error 3 reading from file, row: %d, got:\"%c\"\n",
                    i, carriageReturn);
            fclose(file);
            return;
        }

        char newline;
        if (fscanf(file, "%c", &newline) != 1 || newline != '\n') {
            fprintf(stderr, "Error 4 reading from file, row: %d, got:\"%c\"\n",
                    i, newline);
            fclose(file);
            return;
        }
    }

    fclose(file);

    for (i = 0; i < M_PRED; ++i) {
        for (j = 0; j < N; ++j) {
            // normalized
            t->X[i][j] = data[i][j] / 255.0f;
        }
    }

    freeMat(data, M_PRED);
}
