#filename: CSCI_440-03-m+c_1.4.py
import torch
import torch.nn as nn
import math


#I. Math
#torch.zeros(seq_len,d_model) (?,512)
#https://docs.pytorch.org/docs/stable/generated/torch.zeros.html
#why zeros: to initialize a matrix using integers

#variance
#https://numpy.org/devdocs/reference/generated/numpy.var.html
#why float64, iso (instead of) int, iso (instead) float32
#numpy.var(a)
#calculate measure of spread in an array from the mean
#a: input array
#axis: pick the dimension which to calculate the variance
#dtype: set the datatype for the calc
#ddof: adjust divisor used for variance. ddof=0 calculates the POPULATION variance (denominator: N)
#ddof=1 calculates the SAMPLE variance (denominator: N-1) bc it corrects for bias toward lower values.
#for complex numbers, the absolute value is taken BEFORE squaring so the result is always REAL and nonnegative.	

#math for 2i: even
#i here is index, not imaginary number i
#math for 2i+1: odd
#math for 10k * d_model

#math for sin, code: torch.sin
#math for cos, code: torch.cos
#recall: cos graph is the same as a sine graph, shifted 90 degrees (pi /2) to the left
#sin used for even positions, cos used for odd positions
    #apply the sine to every position, but only the even dimensions
    #apply the sine to even positions, each word will get the sin but only the even dimensions
    #start at 0 and go forward by 2

#math for why log space gives numerical stability, what do i mean by stability: -math.log()
#math for overfit vs underfit

#what's a tensor
#requires_grad_(False)
#show math of exploding variance
#mean0 equation
#variance1 equation

#math.log(x[, base])
#import math
# Natural logarithm (base e) of 10
#ln_value = math.log(10)
#print(f"Natural logarithm of 10: {ln_value}")

# Logarithm of 100 with base 10
#log10_value = math.log(100, 10)
#print(f"Logarithm base 10 of 100: {log10_value}")

# Using the specialized base-10 function
#log10_builtin = math.log10(100)
#print(f"Logarithm base 10 of 100 (builtin): {log10_builtin}")

# Logarithm of 32 with base 2
#log2_value = math.log2(32)
#print(f"Logarithm base 2 of 32: {log2_value}")


#torch.exp(): calculate the EXPONENTIAL of each element in the input tensor: e^x_sub_i. 
#doc: https://docs.pytorch.org/docs/stable/generated/torch.exp.html
#syntax: torch.exp(input, out=None) -> Tensor
#input (mandatory): input tensor containing the values to EXPONENTIATE
#out (optional): output tensor where the result will be stored
#e: Euler's number (2.71828), x_sub_i: an element of the input

#Math Review: EXPONENTIAL
#e^0 ~ 1.0
#e^1 ~ 2.7183 = 2.71828
#e^2 ~ 7.3891
#e^(ln(2)) = 2.0

#example 01: basic use
#import torch
#import math
# Create an input tensor
#x = torch.tensor([0.0, 1.0, 2.0, math.log(2.0)])
# Compute the exponential of each element
#output = torch.exp(x)
#print(output)

#example 02: use the method on a tensor, observe your python distro's precision settings
#import torch
#x = torch.tensor([0.0, 1.0, 2.0])
#Compute the exponential using the tensor method
#output = x.exp()
#print(output)

#are your outputs' float settings similar to
#e^0 ~ 1.0
#e^1 ~ 2.7183 = 2.71828
#e^2 ~ 7.3891
#e^(ln(2)) = 2.0

#example 03: use the method on a tensor, with log of 2
#y_exp0=math.exp(0)
#y_exp1=math.exp(1)
#y_exp2=math.exp(2)
#y=torch.exp(torch.tensor([0, math.log(2.)]))
#output should be: tensor([1., 2.])

#II. Code
#module 2: PositionalEmbeddings/encodings
#original sentence gets mapped to a list of vectors
#want to convey to the model the position of each word inside the sentence
#this is done by adding another vector of the same size as the word embedding (512)
#includes special values given by a formula, that tells the model that this word
#occupies this position in the sentence
#so now we create vectors called Positional Embeddings
#Word Embedding: vector size 512
#Position Embedding: vector size 512. only computed once, and reused for every sentence during training and inference.
#this original word embedding, drives what relative positional frequency is used in the positional embedding for the rest of the future training.

#torch.arange(): returns a 1-D tensor with values from a specified interval, generating a sequence of evenly spaced numbers.
    #torch.arange(start=0, end, step=1, out=None, dtype=None, layout=torch.strided, device=None, requires_grad=False)

    #start (optional): default is 0, staring value for the sequence of points (inclusive)
    #end (mandatory): upper boundary of the interval (exclusive, the value will NOT be included in the tensor)
    #step (optional): the gap between each pair of adjacent points. default=1.
    #dtype (optional): desired datatype of the returned tensor. If None, the dtype is inferred from the start, end, step values.
    #device (optional): device where the tensor will be allocated (e.g. 'cpu', 'cuda' for gpua).
#Try exercises
#>>> import torch
#example 01 default
#>>> torch.arange(5)
#tensor([0, 1, 2, 3, 4])

#example 02 specify start and end
#>>> torch.arange(start=1, end=4)
#tensor([1, 2, 3])

#example 03 specify step size
#>>> torch.arange(start=0, end=10, step=2)
#tensor([0, 2, 4, 6, 8])

#example 03 specify floats
#>>> torch.arange(start=0.2, end=0.5, step=0.1)
#tensor([0.2000, 0.3000, 0.4000])

#example 04 specify dtype
#>>> torch.arange(start=0, end=5, dtype=torch.float32)
#torch.arange(start=0, end=5, dtype=torch.float64)
#tensor([0., 1., 2., 3., 4.])


#torch.exp(), see above

#unsqueeze: to adjust dimensions/shape of tensors.
#https://docs.pytorch.org/docs/stable/generated/torch.unsqueeze.html
#to add 1 new dimension of size 1 at a specified position. Transforms an N-dimensional tensor into an N+1 dimensional tensor.
#it inserts a "singleton" dimension. must specify the dimension or axis where the new dimension should be inserted. helpful for adding a batch 
#dimension to a single data sample for a neural network.
#returns a new tensor w/dimension of size 1 inserted at the specified position
#A key tool in data engineering (hyperparameter updating)
#torch.unsqueeze(input, dim)
#input: the input tensor
#dim: int, the index at which to (insertion point) for the singleton vector
#x = torch.tensor([10, 20, 30, 40])
#print(x)
#t1 = torch.unsqueeze(x, 0)
#print(t1.shape)
#print(t1) #notice the extra pair of brackets, it's a row vector now
#t2 = torch.unsqueeze(x, 1)
#print(t2.shape), it's a col vector now
#print(t2)

class PositionalEncoding(nn.Module):
    #constructor, give it d_model bc it's the size of the vector that the positional encoding needs to be
    #sequence length is the max length of the sentence
    #need to create 1 vector for each position
    #need to give the dropout: to give the model less overfit
    #
    def __init__(self, d_model: int, seq_len: int, dropout: float) -> None:
        super().__init__()
        self.d_model = d_model
        self.seq_len = seq_len
        self.dropout = nn.Dropout(dropout)
    #build shapes (matrix dimensional size manipulation) sequence length of d_model, bc we need vectors of d_model size (512)
    #but we need the sequence length number of them, bc the max length of the sentence is the sequence length
    # Create a MATRIX of shape (seq_len, d_model)
    #pe = positional encoding
    pe = torch.zeros(seq_len, d_model)
    #formulae for pe yt 7:36
    #sentence 1, create a vector of size 512, 1 for each possible position, up to seq_len
    #formula 1 for even positions: PE(0,0) ... PE(0,2)..PE(0,512)
    #formula 1 for even positions: PE(pos,2i) = sin(pos)/(2i/10000*d_model)
    #formula 2 for odd positions: PE(0,1) ..PE(0,511)
    #formula 2 for odd positions: PE(pos,2i+1) = cos(pos)/(2i/10000*d_model)
    #diffs are: formula 1 has 2i vs formula 2 has 2i+1; formula 1 has sin vs formula 2 has cos
    #simplified eqn uses log space, bc it gives numerical stability
    #when you apply an exponential with a log of something you get same number, but more numerically stable

    #create a vector: position, that represents the position of the word inside the sentence
    #Create a VECTOR of shape (seq_len). this goes from 0 to seq_len minus 1
    #create a tensor of shape (seq_len, 1)
    #create numerator of formula 1: pos: position
    position = torch.arange(0, seq_len, dtype=torch.float).unsqueeze(1)

    #create denominator of formula 1: log space 10,000 / d_model
    div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000,0) / d_model))
    #tensor 1 we built: numerator pos: position
    #tensor 2 we built: denominator pos: position. the value will be slightly diff due to change to log space, but result will be the same, and will learn the positional encoding
    #these functions convey positional info to the d_model

    #sin used for even positions, cos used for odd positions
    #apply the sin to every position, but only the even dimensions (bc sin function is symmetric at the origin (0,0))
    #apply the sin to even positions, each word will get the sin but only the even dimensions
    #start at 0 and go forward by 2
    #why use sine and cosine in the first place: seeking numerical address positions that have a regular and unique FREQUENCY
    #trig sine and cosine are REGULAR bc they repeat with the same interval, they can go infinitely in positive/negative direction
    #trig sine and cosine are UNIQUE bc they are shifted by 1 period, they can go infinitely in positive/negative direction
    
    #every position will have the sin, 0, 2, 4, etc.
    pe[:, 0::2] = torch.sin(position * div_term)
    #apply the cos to odd positions,
    #start at 1 and go forward by 2. 1,3,5, etc., but only the odd dimensions (bc cos function is symmetric at y axis (0,1)
    pe[:, 1::2] = torch.cos(position * div_term)

    #add batch dimension to this tensor so we can apply to full sentences with a batch of sentences
    #bc now the shape is seq_length to d_model, to a batch of sentences
    #add a new dimension to this pe, to the first position
    
    pe = pe.unsqueeze(0) * (1, Seq_Len, d_model)

    #stopped here
    #reduce the tensor in the buffer of this module, seq_length in the buffer of d_model
    #when you have a tensor you want to keep inside the module, not as a parameter, learned parameter,
    #but you want it to be saved when you save the file of the model, you should register it as a buffer
    #then the tensor will be saved along w/the state of the buffer
    self.register_buffer('pe', pe)

    #the forward method
def forward(self, x):
    #need to add the positional encoding to every word inside the sentence
    #tell the model we don't want it to learn the positional encoding bc they are fixed,
    #they'll always be the same, they are not learned along the training process
    #this part of the tensor will not be learned: (self.pe[:, :x.shape[1], :]).requires_grad_(False)
    x = x + (self.pe[:, :x.shape[1], :]).requires_grad_(False)
    #apply the dropout
    return self.dropout(x)

