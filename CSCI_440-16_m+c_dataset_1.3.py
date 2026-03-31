# Recall from Lecture: 13
# ->: py arrow symbol, return something per type

#Py magic (dunder) methods
#what are DUNDER methods: Double-UNDERscore methods
#open your terminal
#type: python3
#type: print(dir(int))
#ref: https://www.geeksforgeeks.org/python/dunder-magic-methods-python/ 


#File 03: dataset.py
#II. Code

import torch
import torch.nn as nn
from torch.utils.data import Dataset

#name the dataset: BilingualDataset
class BilingualDataset(Dataset):

    #dataset downloaded from hugginface, tokenizer of src language, tokenizer of tgt language, name of source language, name of target language, sequence length
    def __init__(self, ds, tokenizer_src, tokenizer_tgt, src_lang, tgt_lang, seq_len) -> None:
        super().__init__()

        #save values
        self.seq_len = seq_len
        self.ds = ds
        self.tokenizer_src = tokenizer_src
        self.tokenizer_tgt = tokenizer_tgt
        self.src_lang = src_lang
        self.tgt_lang = tgt_lang

        #we can also save the tokens, the particular tokens we'll use to create the tensors for the model, and special tokens
        #how do we convert token SOS into an Input ID number, there's a tokenizer method for this: .token_to_id()

        #start-of-sentence token, build it into a tensor, w/1 number, either from src or tgt, since both contain these particular tokens
        #the datatype for SOS is int64, why? bc vocabulary lengths can be > 32 bit long
        #repeat for end-of-sentence and padding tokens
        #https://huggingface.co/docs/tokenizers/python/latest/api/reference.html
        self.sos_token = torch.Tensor([tokenizer_src.token_to_id(['[SOS]'])], dtype=torch.int64)
        self.eos_token = torch.Tensor([tokenizer_src.token_to_id(['[EOS]'])], dtype=torch.int64)
        self.pad_token = torch.Tensor([tokenizer_src.token_to_id(['[PAD]'])], dtype=torch.int64)

    #define length method of this dataset, which tells length of the dataset (e.g. huggingface) itself
    def __len__(self):
        return len(self.ds)

    #define get_item dunder method: let a class instance to use the square bracket notation to access elements like a list, tuple, or dictionary
    def __getitem__(self, index: any) -> any:
        #extract original pair from dataset (huggingface)
        src_target_pair = self.ds[index]
        #extract src text and tgt text
        src_text = src_target_pair['translation'][self.src_lang]
        tgt_text = src_target_pair['translation'][self.tgt_lang]

    #convert each text into tokens, and then into Input IDs
    #the Tokenizer will first split the sentence into single words
    #then will map each word to its corresponding number in the vocabulary
    #in 1 path only
    #method .ids = input IDs, gives us IDs corresponding to the original sentence in an array
        enc_input_tokens = self.tokenizer_src.encode(src_text).ids
        dec_input_tokens = self.tokenizer_tgt.encode(tgt_text).ids

        #now pad the sentence to reach the sequence length, bc the model always works with a fixed seq length
        #but we don't have enough words in every sentence, so we use padding token to fill the sentence until it reaches the seq length
        #hence high electricity / compute / storage costs in AI, just to make the math/calculations/code work
        #calculate the number of padding tokens needed for encoder side, decoder side, to reach seq len
        #why -2, bc we have this # of tokens: enc_input_tokens
        #we need to reach this # of tokens: seq_len
        #and we'll add [SOS] and [EOS] tokens, which need to be subtracted out
        enc_num_padding_tokens = self.seq_len - len(enc_input_tokens - 2)
        #for decoder side, we only subtract -1 bc in training we only add SOS token to the Decoder side
        #and then in the label we only add the EOS token, so we only need to subtract -1 to the Decoder side
        dec_num_padding_tokens = self.seq_len - len(dec_input_tokens - 1)

        #ensure the seq len we chose is enough to represent ALL the sentences in our dataset
        #if we choose a seq len too small, we want to raise an exception
        #so the num_padding_tokens should never become negative

        if enc_num_padding_tokens < 0 or dec_num_padding_tokens < 0:
            raise ValueError('Sentence is too long')

        #build the 2 tensors for the encoding and decoding input and the label
        #1 sentence will be sent to the input of the encoder
        #1 sentence will be sent to the input of the decoder
        #1 sentence will be expected output of the decoder, this is called 'label' or 'target'
        #we'll go with 'label' to avoid overloading the term 'target'
        #we concat 3 tensors: SOS token, src text tokens, EOS token, and enough PAD tokens to reach seq len
        #add SOS and EOS to the source text
        encoder_input = torch.cat(
            [
                self.sos_token,
                torch.tensor(enc_input_tokens, dtype=torch.int64),
                self.eos_token,
                #we already calculated the # padding tokens, so add below
                torch.tensor([self.pad_token] * enc_num_padding_tokens, dtype=torch.int64)
            ]
        )

        #Decoder only uses SOS, not EOS
        decoder_input = torch.cat(
            [
                self.sos_token,
                torch.tensor(dec_input_tokens, dtype=torch.int64),
                torch.tensor([self.pad_token] * dec_num_padding_tokens, dtype=torch.int64)
            ]
        )

        #Label only uses EOS (what we expect as output from the decoder)
        label = torch.cat(
            [
                torch.tensor(dec_input_tokens, dtype=torch.int64),
                self.eos_token,
                #we need the same # of padding tokens as the decoder input
                torch.tensor([self.pad_token] * dec_num_padding_tokens, dtype=torch.int64)
            ]
        )

        #Debugging: confirm we reached seg len
        assert encoder_input.size(0) == self.seq_len
        assert decoder_input.size(0) == self.seq_len
        assert label.size(0) == self.seq_len

        #return the tensors so the training can use them
        #return a dictionary comprised of encoder input
        return {
            "encoder_input": encoder_input, #basically Seq_Len # of tokens, added a comma to return multiple arguments 1:45:24
            "decoder_input": decoder_input, #basically Seq_Len # of tokens, added there be a comma here too?
            "encoder_mask": (encoder_input != self.pad_token).unsqueeze(0).unsqueeze(0).int(), # (1, 1, Seq_Len), will be used in self-attention
            "decoder_mask": (decoder_input != self.pad_token).unsqueeze(0).unsqueeze(0).int() & causal_mask(decoder_input.size(0)), # (1, Seq_Len) & (1, Seq_Len, Seq_Len)
            "label": label, # (Seq_Len)
            "src_text": src_text,
            "tgt_text": tgt_text
        }

        #Encoder mask: we increase the size of the encoder input sentence by adding padding tokens
        #but we don't want padding tokens to participate in self attention
        #so we need to build a mask that says that we don't want padding tokens to be seen by the self attention mechanism
        #how do we build the mask, we say all tokens not padding are ok, and all padding tokens are not ok
        #we unsqueeze to add the seq dim and also to add a batch dim, convert to int

        #Decoder mask needs special mask: causal mask: where each word can only look at the previous word
        #and each word can only look at non-padding words, so we don't want padding tokens to participate in self attention
        #and we don't want each word to look at words that come after it, only words that come before it

        #Causal mask needs to build a matrix of size seq_len x seq_len = the size of decoder input
        #1 to Seq_Len combined with 1 to Seq_Len Seq_Len, which can be broadcasted
        #We want each word in the Decoder to only watch words that come before it, in order for it to predict words after it
        #consider a correlation matrix of x: the sentence in order, y: the same sentence in same order
        #thus we want to make all values above the diagonal that represents the multiplication of the query(ies) by the keys in the self-attention mechanism
        #what we want is to hide all of a ith word's future word values so the ith word cannot see those words
        #'your' should not see 'cat is a lovely cat', and only watch itself 'your'
        #'lovely' should see all words before it: 'your cat is a', and not see the word after it: 'cat'
        #solution: mask all values above the diagonal
        #Self Attention in detail:
        #Self Attention is permutation invariant
        #Self Attention requires no parameters. Up to now the interaction between words was driven by their embeddings and positional encodings.
        #this will change later
        #We expect values along the diagonal to be the highest (self correlation is highest)
        #if we do not want some positions to interact, we can set their values to negative infinity before applying the softmax in this matrix
        #so the model will NOT learn those interactions
        #we will use this in the decoder
        #triu: give me every value above the diagonal, want matrix of all 1's
        #this method will return every value above the diagonal and everything below the diagonal will be set to 0
        #everything set to 0 will become TRUE; everything non-zero will become FALSE

        #Label: Seq_Len
        #src text and tgt text for visualization

def causal_mask(size):
    mask = torch.triu(torch.ones(1, size, size), diagonal=1).type(torch.int)
    return mask == 0
