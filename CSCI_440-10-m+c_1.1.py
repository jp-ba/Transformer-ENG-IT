#filename: CSCI_440-10-m+c_1.0.py

#I. Math


#II. Code
#source code: https://github.com/hkproj/pytorch-transformer

import torch
import torch.nn as nn
import math


#Hour 1: Encoder
#module 1: InputEmbeddings
class InputEmbeddings(nn.module)
    #constructor, def dim of model: d_model, vocab_size(# of words in the vocab )
    def __init__(self, d_model: int, vocab_size: int):
        super().__init__()
        #value 1: embedding length
        self.d_model = d_model
        #value 2: vocab size
        self.vocab_size = vocab_size
        #basically a dictionary layer that maps numbers to same vector ele each time, this vector gets learned by the model
        self.embedding = nn.embedding(vocab_size, d_model)

    #forward method: use embedding layer via pytorch to
    def  forward(self, x):
        #paper 3.4: in the embedding layers, we mb the weights by sqrt of d_model
        #input embeddings are now ready
        return self.embedding(x) * math.sqrt(self.d_model)

#module 2: PositionalEmbeddings/encodings
#original sentence gets mapped to a list of vectors
#want to convey to the model the position of each word inside the sentence
#this is done by adding another vector of the same size as the embedding (512)
#includes special values given by a formula, that tells the model that this word
#occupies this position in the sentence
#so now we create vectors called Position Embeddings
#Embedding: vector size 512
#Position Embedding: vector size 512. only computed once, and reused for every sentence during training and inference.

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
    #build shapes sequence length of d_model, bc we need vectors of d_model size (512)
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
    # Create a VECTOR of shape (seq_len). this goes from 0 to seq_len minus 1
    #create a tensor of shape (seq_len, 1)
    #create numerator of formula 1: pos: position
    position = torch.arrange(0, seq_len, dtype=torch.float).unsqueeze(1)

    #create denominator of formula 1: log space 10,000 / d_model
    div_term = torch.exp(torch.arrange(0, d_model, 2).float() * (-math.log(10000,0) / d_model))
    #tensor 1 we built: numerator pos: position
    #tensor 2 we built: denominator pos: position. the value will be slightly diff due to change to log space, but result will be the same, and wil learn the positional encoding
    #these functions convey positional info to the d_model

    #sin used for even positions, cos used for odd positions
    #apply the sin to every position, but only the even dimensions
    #apply the sin to even positions, each word will get the sin but only the even dimensions
    #start at 0 and go forward by 2
    #screen blurry: 10:00
    #every position will have the sin, 0, 2, 4, etc.
    pe[:, 0::2] = torch.sin(position * div_term)
    #apply the cos to odd positions,
    #start at 1 and go forward by 2. 1,3,5, etc.
    pe[:, 1::2] = torch.cos(position * div_term)

    #add batch dimension to this tensor so we can apply to full sentences with a batch of sentences
    #bc now the shape is seq_length to d_model, to a batch of sentences
    #add a new dimension to this pe, to the first position
    #screen blurry: 11:00
    pe = pe.unsqueeze(0) * (1, Seq_Len, d_model)

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

#module 3: layer normalization (aka Add & Norm): for each item in this batch, we calculate a mean and a variance independently from the other items of the batch, then calculate the new values for each of them using their own mean and their own variance
#if you have a batch of n items, each item will have some features
#i.e. each item is a sentence, with many words with its numbers per sentence
#for each item in this batch, we calculate a mean and a variance independently from the other items of the batch
#then calculate the new values for each of them using their own mean and their own variance
#eqn: x hat sub j = (x sub j - mu sub j)/(sqrt var^2 sub j + epsilon)
#parameter: gamma (multiplicative) (sometimes called alpha), mb by each of these x
#parameter: beta (additive) (sometimes called bias), added to each of these x
#used to introduce some fluctuations in the data, bc having all values between
#0 and 1 may be too restrictive for the network.
#the network will learn to tune these 2 params to introduce fluctuations when necessary
#bc we want the model to have the possibility to amplify these values, when it needs to be amplified
#so the model will learn to multiply this gamma by this value in such a way to amplify the values it needs to be amplified

class LayerNormalization(nn.Module):
    #constructor. eps: epsilon, a small number you need to give to the model. 10^-6 (0.000001)
    #we need the epsilon bc: consider the denominator:
    #if variance is 0 , then x hat sub j becomes undefined. #Recall: 200/0 = undefined
    #if variance is very small/very close to zero, then x hat sub j becomes very big.
    #Recall: 200/.1 = 2000 very big. then the x hat sub j would get very big.
    #x hat sub j getting very big is undesirable bc the cpu/gpu can only represent
    #numbers up to a certain position and scale, so we don't want
    #very big numbers or very small numbers
    #so for numerical stability, we use epsilon, also to avoid division by 0
    def __init__(self, eps: float = 10**-6) -> None:
        super().__init__()
        #save the epsilon
        self.eps = eps
        #alpha (multiplication); bias (additive)
        #nn.Parameter makes the param learnable
        self.alpha = nn.Parameter(torch.ones(1))    #multiplied
        self.bias = nn.Parameter(torch.zeros(1))    #added

    def forward(self, x):
        #-1 bc everything after the batch
        #keepdim bc the mean usually cancels the dimension to which it's applied
        mean = x.mean(dim * -1, keepdim=True)
        std = x.std(dim * -1, keepdim=True)
        return self.alpha * (x - mean) / (std + self.eps) + self.bias

#module / layer 4: feed forward
    #a fully connected layer, that's used in the encoder and the decoder
    #the paper: section 3.3, FFN: Feed Forward Network
    #it is 2 matrices: W1 and W2, mb x with ReLu in between
    #b1 is the added bias
    #can do this in pytorch with linear layer
    #where we define 1st one as W1 + b1, and W2 + b2
    #and in between we apply ReLu (the max)
    #1st one is d model=512 to dff=2048 and back

class FeedForwardBlock(nn.Module):
    #constructor
    def __init__(self, d_model: int, d_ff: int, dropout: float) -> None:
        super().__init__()
        #matrix 1
        self.linear_1 = nn.Linear(d_model, d_ff)    #W1 and B1
        self.dropout = nn.Dropout(dropout)
        self.linear_2 = nn.Linear(d_ff, d_model)    #W2 and B2

    def forward(self, x):
        #?fuzzy character: input sentence
        #(Batch, Seq_Len, d_model) convert it using Linear 1, into another tensor Batch, Seq_Len, d_ff
        #bc if we apply this Linear d_model into d_ff
        #then we apply linear to convert it back to d_model
        ? (Batch, Seq_Len, d_model) --> (Batch, Seq_Len, d_ff) --> (Batch, Seq_Len, d_model)
        return self.linear_2(self.dropout(torch.relu(self.linear_1(x))))

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

#0:24:22
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

    #staticmethod: means you can call fx attention w/o having an instance of this class: MultiHeadAttention
    def attention(query, key, value, mask, dropout: nn.Dropout):
        #d_k is the last dim of the query key and the value
        d_k = query.shape[-1]
        #1st part of formula: query mb transpose of key, db sqrt of d_k
        #blurry: pytorch matrix multiplication xi
        #-2, -1 means transpose the last 2 dimensions. last dim seq_len, d_k becomes d_k, seq_len

        # (Batch, h, Seq_Len, d_k) --> (Batch, h, Seq_Len, Seq_Len)
        attention_scores = (query ?* key.transpose(-2, -1)) / math.sqrt(d_k)
        #before we apply softmax, we need to apply the mask to hide interactions btwn some words
        #so we apply the mask, then apply the softmax
        #the softmax will take care of the values that we replaced
        #how do we apply the mask: for all the values we want to mask
        #we just replace them with very small values so softmax will replace them with 0
        if mask is not None:
            attention_scores.masked_fill_(mask == 0, -1e9)
#mask == 0, -1e9 means: replace all values for which mask == 0 is true, with value: -1e9
#all the values we don't want in the attention: some words to watch future words when we build decoder
#we don't want the padding values to interact with other values bc they're filler words to reach seq leng
#we'll want to replace them with -1e9, which represents -infinity
        attention_scores = attention_scores.softmax(dim = -1)   # (Batch, h, seq_len, seq_len)
        #when we apply softmax, it will replace with zero, we apply it to this dim
        #we mb the output of the softmax by V matrix
        if dropout is not None:
                attention_scores = dropout(attention_scores)
        return (attention_scores * value), attention_scores #return the attention_scores mb V matrix and attention_scores themselves
        #why do we return a tuple? we need 'attention_scores * value' for the model to give it to the next layer
        #'attention_scores' will be used for visualization, the output of the multihead attention will be here
        #we'll visualize the score given by the model for that particular interaction


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
        #now we need to calculate Attn, need to make a function to calculate, between: self.dropout = nn.Dropout(dropout) & def forward(self, q, k, v, mask): staticmethod

        #we want output and attention scores, output of the softmax
        x, self.attention_scores * MultiHeadAttentionBlock.attention(query, key, value, mask, self.dropout)

        #we calculated softmax (Q K^T)/(sqrt (d_k)) x V, which => smaller green matrices, h1 .. hn
        #now we need to concat them to make Matrix H, then mb Woutput
        #we transpose bc before we transformed the matrix into seq_len,
        #we had seq_len as 3rd dim and we want it back into 1st base to combine them
        #bc the resulting tensor, we want seq_len to be in 2nd position
        # (Batch, h, Seq_Len, d_k) -->  (Batch, Seq_Len, h, d_k) -->  (Batch, Seq_Len, d_model)
        #a view won't work here
        #contiguous lets pytorch to transform the shape of a tensor needs to be put into memeory to be contiguous, so it can do it in place
        x = x.transpose(1, 2).contiguous().view(x.shape[0], -1, self.h * self.d_k)
        #self.h * self.d_k is d_model
        #finally multiply x by Wo
        # (Batch, Seq_Len, d_model) --> (Batch, Seq_Len, d_model)
        return self.w_o(x)

        #The output of left side Encoder's 1st Add & Norm, goes to Feed Forward and 2nd Add & Norm
        #The output of Feed Forward also goes to 2nd Add & Norm.  and combined together at the 2nd Add & Norm
        #The output after Positional Encoding before it goes to Multi Head Attention, goes to Multi Head Attn and then skips 1 layer (Multi Head Attn) and goes to 1st Add & Norm
        #the 1st Add & Norm combines the skipped layer (Multi Head Attention) and the output of Multi Head Attention
        #we call this layer we build a 'Residual Connection' bc it's a skipped connection
        class ResidualConnection(nn.Module):

            def __init__(self, dropout: float) -> None:
                super().__init__()
                self.dropout = nn.Dropout(dropout)
                #the skipped connection is between the Add&Norm and the previous layer, so we need the norm
                self.norm = LayerNormalization()

        #sublayer: the previous layer. we combine x with output of previous layer
        #this the def of Add & Norm
            def forward(self, x, sublayer):
                #this deviates from the paper bc other implementations did it this way
                return x + self.dropout(sublayer(self.norm(x)))

        #the Encoder block is Multi Head Attention + Add&Norm1 + Feed Forward + Add&Norm 2. This combined block can be repeated/cloned Nx times
        #where the output of the combined previous Encoder block can be sent to the next one
        #and the output of the final Encoder block is sent to the Decoder block
        #The Encoder block has 2 subblocks, and the Decoder block has 3 subblocks

        #the 'self' is the multiheaded self attention
        #we call it self_attention bc in the case of the Encoder,
        #it is applied to the same input w/3 different roles: query, key, value
        #which is our FeedForwardBlock

        class EncoderBlock(nn.Module):

            def __init__(self, self_attention_block: MultiHeadAttentionBlock, feed_forward_block: FeedForwardBlock, dropout: float) -> None:
                super().__init__()
                self.self_attention_block = self_attention_block
                self.feed_forward_block = feed_forward_block
                #moduleList is a way to organize a list modules, in this case 2
                #define our 2 residual connections
                self.residual_connections = nn.ModuleList([ResidualConnection(dropout) for _ in range(2)])

            #define the source mask: src_mask: the mask we want to apply to the input encoder
            #why need it: we want to hide interaction of padding words with other words

            def forward(self, x, src_mask):
                #1st skip connection is between pe and the 3 part fork before Multi Head Attention (MHA)
                #the x gets sent to 3 part fork Multi Head Attention and 1st Add&Norm (A&N)
                #the 1st x and the 2nd x (lambda x) from MHA output combine at A&N1
                #we define the sublayer lambda x, we first apply the self attention in which we give the query
                #it's called self attention bc: we give the query, key, value is over x our input
                #bc the role of the query, key, value is x itself,
                #so the sentence is watching itself: 1 word of a sentence is interacting with other words of the same sentence
                #this differs from the Decoder bc of cross-attention
                #query coming from Decoder are watching the key and the values coming from the Encoder
                #the self_attention_block (SAB) is calling the def forward(self, q, k, v, mask) fx of the MHA block
                #so we give the SAB, query, key, value and the src_mask
                #this will be combined with the x by using the ResidualConnection
                #this is the residual_connection 1 (RC1)
                x = self.residual_connections[0](x, lambda x: self.self_attention_block(x, x, x, src_mask))
                #this is the residual_connection 2 (RC2): combine output of feed_forward_block with x from RC1
                x = self.residual_connections[1](x, self.feed_forward_block)
                #this defines our Encoder block
                return x
        #Now define Encoder objects, can have up to N of them

        class Encoder(nn.Module):
            # number of layers = n from ModuleList
            def __init__(self, layers: nn.ModuleList) -> None:
                super().__init__()
                self.layers = layers
                self.norm = LayerNormalization()

            def forward(self, x, mask):
                #apply 1 layer after another
                for layer in self.layers:
                    x = layer(x, mask)
                    #the output of the previous layer becomes input for next layer
                    #finally apply the normalization
                return self.norm(x)
            #0:51 this concludes Encoder portion
            #Skip Connections: each side arrow that bifurcates from MHA and goes to A&N1 along w/ FeedForward (FF) to A&N2 are each 'x'
#Hour 2: Decoder. 0:52:00

#Output Embedding are same as Input Embeddings, the class is the same, we initialize it twice
#PE can use same values for the Decoder as we use for the Encoder
#the 3rd block in the Decoder block is Masked Multi Headed Attention (MMHA) along with Skip Connections
#The Encoder block has 2 sublayers: MHA, FF, 2 SC
#The Decoder block has 3 sublayers: MMHA, MHA, FF, 3 SC

class DecoderBlock(nn.Module):

#the self attention here is the MMHA, not the MHA
#bc the input is used 3x: q, k, v and the same input is used as the q,k,v
#which means each word in the sentence is matched with each other word in the same sentence
#but sublayer 2: MHA, we'll have attention calcuated using query coming from the Decoder
#while the key and values will come from the Encoder
#thus the MHA here is not self-attention, it's called a cross-attention
#it's crossing 2 diff objects together and matching them to calculate the relationship btwn them

    def __init__(self, self_attention_block: MultiHeadAttentionBlock, cross_attention_block: MultiHeadAttentionBlock, feed_forward_block: FeedForwardBlock, dropout: float) -> None:
        super().__init__()
        self.self_attention_block = self_attention_block
        self.cross_attention_block = cross_attention_block
        self.feed_forward_block = feed_forward_block
        #3 Residual Connections (RC)
        self.residual_connections = nn.ModuleList([ResidualConnection(dropout) for _ in range(3)])

#x: input of the Decoder
#encoder_output
#src_mask: applied to Encoder, source language
#tgt_mask: target mask applied to Decoder, target language
#we have src_mask and tgt_mask bc we're doing a translation task from source language to target language
    def forward(self, x, encoder_output, src_mask, tgt_mask):
        #call the 1st RC[0]
        #calc self attention
        #x, x, x are q, k, v.
        #q, k, v are same input, but with mask of Decoder, bc this is the self_attention_block of the Decoder
        x = self.residual_connections[0](x, lambda x: self.self_attention_block(x, x, x, tgt_mask))
        #calc cross attention, the 2nd RC[1]. q comes from Decoder, k and v come from Encoder, mask of Encoder
        x = self.residual_connections[1](x, lambda x: self.cross_attention_block(x, encoder_output, encoder_output, src_mask))
        #calc the 3rd RC[2], the FF block
        x = self.residual_connections[2](x, self.feed_forward_block)
        return x
        #this is final component to build a Decoder now, just a matter of # of Decoder blocks, Nx

class Decoder(nn.Module):

    def __init__(self, layers: nn.ModuleList) -> None:
        super().__init__()
        #normalization
        self.layers = layers
        self.norm = LayerNormalization()

    def forward(self, x, encoder_output, src_mask, tgt_mask):
        #apply input to 1 layer, use the output of the previous layer as input of next layer
        for layer in self.layers:
            x = layer(x, encoder_output, src_mask, tgt_mask)
            #we're calling the forward method here: def forward(self, x, encoder_output, src_mask, tgt_mask):
        return self.norm(x)