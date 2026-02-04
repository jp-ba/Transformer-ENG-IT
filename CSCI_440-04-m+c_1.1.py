#filename: CSCI_440-04-m+c_1.0.py

#I. Math
#The layer normalization equation with epsilon (\(\epsilon \)) is used to stabilize training by normalizing the inputs across the feature dimension for each sample in a batch. 

#x sub i : The input feature vector.
#mu: The mean of the input features.
#sigma^2: The variance of the input features.
#epsilon: A small constant (e.g., \(10^{-5}\) or \(10^{-8}\)) added to the variance to improve numerical stability and prevent division by zero.
#gamma: A learnable scaling parameter (initialized to 1).
#beta: A learnable offset/bias parameter (initialized to 0). 

#Independence: Unlike Batch Normalization, Layer Normalization computes the mean and variance independently for each training sample.
#Application: It is commonly used in Recurrent Neural Networks (RNNs) and Transformers.
#Numerical Stability: The addition of \(\epsilon \) ensures that even if the variance \(\sigma ^{2}\) is zero, the division operation does not fail. 
                                                                                                                                                                                
#II. Code
#module 3: layer normalization (aka Add & Norm): for each item in this batch, 
# we calculate a mean and a variance independently from the other items of the batch, 
# then calculate the new values for each of them using their own mean and their own variance
#if you have a batch of n items, each item will have some features
#i.e. each item is a sentence, with many words with its numbers per sentence
#for each item in this batch, we calculate a mean and a variance independently from the other items of the batch
#then calculate the new values for each of them using their own mean and their own variance
#eqn: x hat sub i = (x sub i - mu sub)/(sqrt var^2 + epsilon)
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