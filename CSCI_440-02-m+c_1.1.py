#filename: CSCI_440-02-m+c_1.0.py

#I. Math
import math

#https://docs.python.org/3/library/math.html#math.sqrt
#math.sqrt(x)

#what happens when you calculate square root of radicand: integers? answer (square root) gets smaller
a_int = math.sqrt(25)

#what happens when you calculate square root of radicand: decimals? answer (square root) gets bigger
a_float = math.sqrt(.25)

#what happens when you calculate square root of radicand: log? answer rewrite square root as exponent, multiply by fraction. 
#goes from calculating the "root of a log" to a "log of a root"
#log(x^(1/2)) = (1/2) * log(x)
a_log = math.sqrt(math.log(25))

#II. Code

import torch
#https://docs.pytorch.org/docs/stable/torch.html
#package: torch 
#Note: a 'package' is a collection of 'modules'.


import torch.nn as nn
#https://docs.pytorch.org/docs/stable/nn.html
#module: nn, alias it using 'as nn'
#contains all the building blocks necessary to design and train neural networks
"""
Layers: It includes pre-defined layers like nn.Linear (fully connected layer), nn.Conv2d (convolutional layer), and nn.MaxPool2d (pooling layer).

Activation Functions: While many activation functions (like ReLU) are often used from the functional module (torch.nn.functional typically imported as F), they are essential components for neural networks.

Loss Functions: It offers various loss functions for optimizing models.
nn.Module Base Class: All neural network models and layers in PyTorch subclass the nn.Module class, allowing for parameters to be managed, moved to the GPU, and saved easily. 
"""

import math
#https://docs.python.org/3/library/math.html#module-math
#This module provides access to the mathematical functions defined by the C standard.

#module 1: InputEmbeddings
class InputEmbeddings(nn.module):
    #constructor, def dim of model: d_model, vocab_size(# of words in the vocab )
    def __init__(self, d_model: int, vocab_size: int):
        #The __init__ method in Python is a special initializer method for classes. 
        #It is automatically called every time a new instance (object) of a class is created. 
        #Its primary purpose is to set the initial state and attributes of the new object. 
        #https://docs.python.org/3/library/functions.html#super
        #class super(type, object_or_type=None, /)
        super().__init__()
        #value 1: embedding length
        self.d_model = d_model
        #value 2: vocab size
        self.vocab_size = vocab_size
        #basically a dictionary layer that maps numbers to same vector ele each time, this vector gets learned by the model
                #https://docs.pytorch.org/docs/stable/generated/torch.nn.modules.sparse.Embedding.html
        self.embedding = nn.embedding(vocab_size, d_model)
        #nn.embedding: module in PyTorch is a lookup table that stores a fixed-size dictionary of embeddings. 
        #It is widely used in Natural Language Processing (NLP) to convert integer indices (representing words or tokens) into dense, continuous vectors. 
        #Purpose: It maps discrete, high-dimensional integer inputs (like vocabulary indices) to continuous, lower-dimensional dense vector representations.
        #Mechanism: Internally, it is an M x N weight matrix, where M is the number of unique items (vocabulary size) and N is the size of each embedding vector (embedding dimension). 
        # The input indices are used to select specific rows from this matrix.Trainable: The embedding vectors start as randomly initialized floating-point values and are updated via backpropagation during the training process, allowing the model to learn meaningful representations (e.g., words with similar meanings having similar vectors).Efficiency: It is a more efficient alternative to a fully connected linear layer when dealing with a large number of input features (a large vocabulary). 

    #forward method: use embedding layer via pytorch to
    #https://docs.pytorch.org/docs/stable/generated/torch.nn.modules.sparse.Embedding.html
    def  forward(self, x):
        #paper 3.4: in the embedding layers, we mb: 'multiply by' the weights by sqrt of d_model
        #input embeddings are now ready
        #math.sqrt(x): here we take the square root of self.d_model
        return self.embedding(x) * math.sqrt(self.d_model)
