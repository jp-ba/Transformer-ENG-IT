#filename: CSCI_440-07-m+c_1.0.py

#I. Math

#review: randn
#https://docs.pytorch.org/docs/stable/generated/torch.randn.html
torch.randn(4)
torch.randn(2, 3)

#review: Dropout
#https://docs.pytorch.org/docs/stable/generated/torch.nn.Dropout.html
m = nn.Dropout(p=0.2)
input = torch.randn(20, 16)
output = m(input)

#ReLU
import numpy as np

def relu(x):
    """
    Applies the Rectified Linear Unit (ReLU) activation function.

    Parameters:
    x (numpy.ndarray or array-like): The input array.

    Returns:
    numpy.ndarray: The output array with negative values replaced by zero.
    """
    return np.maximum(0, x)

# Create a sample NumPy array with positive and negative values
data = np.array([-1.0, 4.1, 0.0, 5.0, -2.8, 6.5, -7.3, 8.0])

# Apply the ReLU function
output = relu(data)

print("Input  array:", data)
print("Output array:", output)

# Example
x = np.array([-3, -1, 0, 2, 5])
print(relu(x))

# Output: [0 0 0 2 5]

#II. Code
#module / block 5: Multi-Head Attention
     #Encoder: multi head attention takes input of the encoder and uses it 3x
     #1: query: q
     #2: key: k
     #3: values: v
     #encoder: the input sequence is (seq_length, d_model),
     #transform into 3 matrices which are exactly the same as the input bc we're talking about the encoder (it's diff for decoder)
     #matrix v (seq, d_model) * matrix w (d_model, d_model) = v (seq, d_model)
     #then split this matrix into individual h matrices (the number of heads for the # of heads)
     #split the matrices along the embedding dimensions, not along the sequence dimension
     #which means that each head will have access to the full sentence
     #but a different part of the embedding of each word
     #we apply the Attention to each of the smaller matrices using this formula
     # Attention(Q,K,V) = softmax (QK^T)/(sqrt(d sub k)) * V
     #head sub i = Attention(QW sub i ^Q, KW sub i ^K, VW sub i ^V)
     #this will give us smaller green Head matrices
     #then we combine them back by concatenating
     #seq = sequence length
     #d_model = size of the embedding vector
     #h = # of heads
     #d sub k -- d sub v = d_model / h
     #then we combine them back by concatenating: MultiHead(Q,K,V) = Concat(head sub 1 ... head sub h) * W^O
     #H x W^O = MultiHeadAttn output
     #MultiHeadAttn output matrix has same dim as the input matrix
     #the output of MultiHeadAttn is also seq x d_model
     #this slide doesn't show batch dimension bc we're talking about 1 sentence
     #but when we code a transformer, we don't work with 1 sentence, but mltpl sentences
     #keep in mind there's another dimension here, that's a batch dimension

class MultiHeadAttentionBlock(nn.Module):
    #h = # of heads
    def __init__(self, d_model: int, h: int, dropout: float) -> None:
        super().__init__()
        #save the values
        self.d_model = d_model
        self.h = h
        #we have to divide the embedding v into h heads
        #which means the d_model must be divisible by h
        #otherwise, ?d_model we cannot divide equally the same vector that
        #represents the embedding into equal matrices for each head
        #so we make sure d_model is divisible by h
        assert d_model % h == 0, "d_model is not divisible by h"
        #d_model db h = Dk. (d_model / h) = Dk
        #if we divide d_model by h heads, we get new value: dk
        self.d_k = d_model // h
        #define matrix Wq. d_model x d_model, so output will be seq x d_model
        self.w_q = nn.Linear(d_model, d_model)  # Wq
        self.w_k = nn.Linear(d_model, d_model)  # Wk
        self.w_v = nn.Linear(d_model, d_model)  # Wv
        #output matrix: Wo for output. h x Dv x d_model. since h x Dv = d_model, then it's d_model x d_model
        #h x dv = dk bc it's d_model db h
        #head 1, when you look at the Attn eqn, the softmax is mb V
        #in the paper, this value is dv
        #dv = dk (on practical level)
        #27:10 fuzzy the subscripts in the slide boxes
        self.w_o = nn.Linear(d_model, d_model)  #Wo
        self.dropout = nn.Dropout(dropout)

#stopped here lecture 06
        #mask: if we want some words to not interact with other words, we mask them
        #when we calculate eqn: Attn(Q,K,V), we get green Head matrix
        #0:28:48 but before we mb V, the softmax((QK^T)/(sqrt(d sub k))), produces this orange matrix, each word by each OTHER word, a seq by seq matrix
        #if we don't want some words to interact with other words, we replace the value, the attention score with something that is very small before we apply the softmax
        #and then we apply the softmax, then these values become zero
        #bc as you recall, the softmax on the numerator has e^x, so if x goes to -infinity, so very small number, e^-infinity will become very close to zero
        #basically, this hides the attention for those 2 words, this is the job of the mask
    def forward(self, q, k, v, mask):
        #query mb Wq, this produces matrix q prime
        query = self.w_q(q) # (Batch, Seq_Len, d_model) --> (Batch, Seq_Len, d_model), gives us same dim as original Q matrix
        key = self.w_k(k)   # (Batch, Seq_Len, d_model) --> (Batch, Seq_Len, d_model), gives us same dim as original K matrix
        value = self.w_v(v) # (Batch, Seq_Len, d_model) --> (Batch, Seq_Len, d_model), gives us same dim as original V matrix

        #we want to divide these query, key, value matrices into smaller matrices, so we can give each small matrix to a different head
        #use view method of pytorch which means we keep the batch dimension, we don't want to split the sentence, we want to split the embedding into h parts
        #we also want to keep the 2nd dimension, which is the sequence, bc we don't want to split it
        #the third dimension, the d_model, we want to split it to two smaller dimensions: which is h by dk
        #recall: dk is d_model db h, so d_k mb h gives you back d_model, now transpose
        # (Batch, Seq_Len, d_model) --> (Batch, Seq_Len, h, d_k) --> transpose (Batch, h, Seq_Len, d_k)
        query = query.view(query.shape[0], query.shape[1], self.h, self.d_k).transpose(1, 2)
        #reason we transpose: bc we prefer to have h dimension instead of being the 3rd dim, to be the 2nd dim
        #this way each view head will see all the sentence, so it will see this dim query.shape[1] seq length by dk
        #important bc we want each head to watch: Seq_Len, d_k
        #so each head will see the full sentence i.e. each word in the sentence, but only a smaller part of the embedding
        #repeat for the key and value
        key = key.view(key.shape[0], key.shape[1], self.h, self.d_k).transpose(1, 2)
        value = value.view(value.shape[0], value.shape[1], self.h, self.d_k).transpose(1, 2)
        #we have now split the Q, K, V matrices into smaller matrices


