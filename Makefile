
CC=gcc
CFLAGS=-g -Wall -Werror -pedantic -Iinclude

SRC_FILES=src/utils.c src/data.c src/ann.c

all: $(SRC_FILES) src/train.c src/predict.c 
	$(CC) $(CFLAGS) -o train $(SRC_FILES) src/train.c -lm
	$(CC) $(CFLAGS) -o predict $(SRC_FILES) src/predict.c -lm

clean:
	rm predict train

