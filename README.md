# Deep Learning Neural Networks

This is a project that implementing Multilayer Perceptron and Convolution Neural Networks without DL library(e.g. tensorflow,pythorch......)
And we use MNIST dataset to demo our model. 
## Getting Started

You don't need install any other package but Python 3.5 only. 

### Prerequisites

Anaconda will be a pretty convenient tool to install any version of Python. Please check the url below and download as your requested.

```
https://www.anaconda.com/download/
```

### Installing

First, before running demo.ipynb, you should acquire the dataset first, which is in the ./dataset/mnist_dataset	

```
unzip dataset/mnist_dataset.zip
```

Second, to visualize and analysis the outcome, we take adventage of Jupyter Notebook. It would be installed when you install python.
If it's not in your package pool, try pip install.  

```
$ pip3 install --upgrade pip

$ pip3 install jupyter
```

## Running the tests

There are four main targets in the implementation:

- Implement a Convolutional Neural Network (CNN) that consists of convolutional layers and fully-connected (FC or MLP) layers
- Instantiate a [784, 50, 10] MLP and a LeNet-5-like CNN
- Train both MLP and CNN using the MNIST dataset
- Exploit the impact of hyperparameters (learning rate, batch size, activation function, etc) on training accuracy and training time

Running demo.ipynb 

```
Just execute each cell of demo.ipynb to get the result.
```

## Authors

The project is the assignment of the course, Programming Top Ten Important Algorithms in Python, lectured by Professor Youn-Long Lin, C.S. NTHU 

Details implemented by Min-Han Tsai, C.S. NTHU
