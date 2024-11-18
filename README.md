# MNIST Digit Classification with Dense Neural Network

This repository contains a simple implementation of a neural network for classifying MNIST handwritten digits. The model is trained using Keras and TensorFlow, with a basic dense layer architecture. The project is based on the first chapter of the book *Deep Learning with TensorFlow and Keras*.

## Repository

- **GitHub Profile**: [Wajahat-Infinity](https://github.com/Wajahat-Infinity)
- **Repository**: [mnist-digit-classification](https://github.com/Wajahat-Infinity/mnist-digit-classification)

## Project Overview

The objective of this project is to classify the MNIST dataset, which consists of 70,000 grayscale images of handwritten digits (0-9), into one of 10 classes.

The model is trained with the following parameters:

- **Epochs**: 200
- **Batch Size**: 128
- **Validation Split**: 0.2
- **Neurons in Hidden Layer**: 128
- **Optimizer**: Stochastic Gradient Descent (SGD)
- **Loss Function**: Categorical Cross-Entropy
- **Activation Function**: Softmax

The architecture is a simple fully connected neural network with one dense layer.

## Requirements

To run this project, install the required dependencies using the following command:

```bash
pip install tensorflow keras numpy matplotlib
```
## Dataset

The MNIST dataset is a collection of 60,000 training images and 10,000 testing images of handwritten digits, available at [MNIST dataset](http://yann.lecun.com/exdb/mnist/).

## Model Architecture

The model consists of the following layers:

- **Input Layer**: Reshaped image data from the MNIST dataset.  
- **Dense Layer**: A fully connected layer with 128 neurons and softmax activation.  
- **Output Layer**: 10 neurons corresponding to the digits 0-9.  

## Training

The model is trained for 200 epochs with a batch size of 128. A validation split of 20% is used to monitor the model's performance during training.

## Results

- **Validation Accuracy**: 92.3%  
- **Test Accuracy**: 92.2%  

## How to Use

1. Clone this repository:

```bash
   git clone https://github.com/Wajahat-Infinity/mnist-digit-classification.git
```


2. Navigate to the project directory:


3. Open and run the Jupyter notebook to train the model:

   ```bash
   jupyter notebook mnist_digit_classification.ipynb
   cd mnist-digit-classification
   ```
   The trained model is saved as `mnist_dense_model.h5`. Load it for inference:

```python
from keras.models import load_model
model = load_model('mnist_dense_model.h5')
```
## Acknowledgments

- The MNIST dataset is provided by [Yann LeCun](http://yann.lecun.com/exdb/mnist/).
- Model architecture and training process are based on *Deep Learning with TensorFlow and Keras*.
