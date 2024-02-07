# mnist-ann-c
A simple artificial neural network to classify handwritten digits using [MNIST dataset](https://en.wikipedia.org/wiki/MNIST_database). Achieves ~85% accuracy.

## Data
The data used can be found [here](https://www.kaggle.com/datasets/macsencasaus/digital-recognizer-no-header/data?select=train.csv). This data is from
[Digital Recognizer](https://www.kaggle.com/competitions/digit-recognizer/data) just without the the csv headers.

Place the `train.csv` in the `data/train` directory and the `test.csv` in the `data/test` directory.

## Getting Started
To build, just run make!
```
make
```
This will compile the code into two binaries: `train` and `predict`.

Run
```
./train
```
to train the model on the training data based on the default parameters. 

Then run
```
./predict <index>
```
with some 0 <= index <= 999 to apply the model to one sample in the testing data. It will display the image being tested, the model's prediction, and the model's confidence.

## Advanced
You can modify the parameters found in [include/parameters.h](https://github.com/macsencasaus/mnist-ann-c/tree/main/include/parameters.h) to change the characteristics and accuracy of the model!

## Resources
- https://www.youtube.com/watch?v=w8yWXqWQYmU&t=530s&ab_channel=SamsonZhang
