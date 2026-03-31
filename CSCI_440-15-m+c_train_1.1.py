#filename: CSCI_440-16-m+c_1.0.py

#I. Math


#II. Code
#the following code is for train_1.0.py file

#dataset: https://huggingface.co/datasets/opus_books/viewer/en-it/train?row=60
#this is only library we'll use besides pytorch and huggingface tokenizer library to transform this text into a vocabulary
#subset: en-it
#each data item has id (string), translation (a pair of sentences in { "src": "source blah", "tgt": "tgt blah" }
#we'll train our transformer to translate from src language to tgt language
#make code to dl this dataset, create tokenizer
#tokenizer: comes before Input Embedding
#tokenizer goal: split the sentence into single words (many strategies: bpe tokenizer, word level tokenizer, sub-word level, word part)
#we'll use simplest: word level tokenizer: splits sentences by whitespace as word boundaries
#then map each word to 1 number, to build a vocabulary
#in the process we can build special tokens which will be used for the Transformer: padding, SOS: start-of-sentence, EOS: end-of-sentence, necessary for training the Transformer

#get video explaining what a tensor is: https://www.youtube.com/watch?v=1GwAEnegaRs&list=PL5rMsAP3NitHpWUFnldRn95OvOlah6Wj9&index=13

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split

#import dataset and causal_mask from dataset.py
from dataset import BilingualDataset, causal_mask

#import model
from model import build_transformer

#import weights via config.py preload when training crashes
from config import get_weights_file_path, get_config

#get this from pip
from datasets import load_dataset
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
#the class that will train the tokenizer, create the vocabulary given the list of sentences
from tokenizers.trainers import WorldLevelTrainer
#split words by whitespace
from tokenizers.pre_tokenizers import Whitespace

#import TensorBoard
from torch.utils.tensorboard import SummaryWriter

#import warnings, many will come from cuda
import warnings

#import tqdm (progress bar for training loop)
from tqdm import tqdm


#ability to create absolute paths from relative paths
from pathlib import Path

#run greedy decoding on our model, run the encoder only once
def greedy_decode(model, source, source_mask, tokenizer_src, tokenizer_tgt, max_len, device):
    sos_idx = tokenizer_tgt.token_to_id('[SOS]')
    eos_idx = tokenizer_tgt.token_to_id('[EOS]')

    #Precompute the encoder outut and reuse it for every token we get from the decoder
    #give it the source and source_mask, which is the encoder input and the encoder mask
    encoder_output = model.encode(source, source_mask)

    #how do we inference:
    #step 1: we give to the decoder the SOS token, so the decoder will output the 1st token of the translated sentence
    #step 2: at every iteration, we add the previous token to the Decoder input, so that the decoder can output the next token
    #step 3: we take the next token and put it again as input to the Decoder to get the successive token
    #build the decoder input for the 1st iteration, the SOS token
    #initialize the decoder input with the SOS token
    decoder_input = torch.empty(1,1).fill_(sos_idx).type_as(source).to(device)

    #we'll keep asking the Decoder to output the next token until either a) we reach EOS token, b) the max_len defined in fx's argument above
    #stopped 2:28
    while True:
        #1st stopping condition: if the decoder output, which becomes input of next step, reaches max_len
        #why do we have 2 dimensions: empty(1,1), 1 is for the batch, 1 is for the tokens of the decoder input
        if decoder_input.size(1) == max_len:
            break

        #build mask for the tgt (decoder input)
        #use causal_mask to prevent the decoder input from looking at future words
        #we don't need the other mask b/c here we don't have any PAD tokens
        decoder_mask = causal_mask(decoder_input.size(1)).type_as(source_mask).to(device)

        #calculate the output of the decoder. reuse the output of the encoder for every iteration of the loop
        out = model.decode(encoder_output, source_mask, decoder_input, decoder_mask)

        #Get the next token, get the probabilities of the next token using the projection layer
        #but we only want the projection of the last token: -1
        #the next token after the last we have given the encoder
        prob = model.project(out[:,-1])
        #Select the token with the maximum probability (using the greedy search)
        _, next_word = torch.max(prob, dim=1)
        #take next_word and append it back to decoder_input, bc it will be the input of the next iteration
        decoder_input = torch.cat([decoder_input, torch.empty(1,1).type_as(source).fill_(next_word.item()).to(device)], dim=1)

        #if next word is equal to the End Of Sentence token, then stop
        if next_word == eos_idx:
            break

    #return the output, which is basically the decoder_input, bc everytime you're appending the next token to it
    #we remove the batch dim, so we squeeze it
    return decoder_input.squeeze(0)

def run_validation(model, validation_ds, tokenizer_src, tokenizer_tgt, max_len, device, print_msg, global_state, writer, num_examples=2):
    #num_examples=2 means we'll sample 2 sentences
    #place model into evaluation mode, this tells pytorch we'll evaluate the model
    model.eval()
    count = 0

    #create placeholder storage lists
    #
    source_texts = []
    expected = []
    predicted = []

    #size of the control window (just use a default value)
    console_width = 80

    #disables gradient calculation for every tensor we'll run inside this 'with' block
    #this is what we want, we just want to inference from the model, not train it during this loop
    with torch.no_grad():
        #get a batch from this validation dataset
        for batch in validation_ds:
            #keep a count of how many we processed
            count += 1
            #get input from this current batch, the validation_ds, we only have a batch size of 1
            #this is
            encoder_input = batch['encoder_input'].to(device)
            encoder_mask = batch['encoder_mask'].to(device)

            #verify the batch size is 1
            assert encoder_input.size(0) == 1, "Batch size must be 1 for visualization"
            #when we want to inference the model, we need to calculate the encoder input only once
            #and reuse it for every token the model will output from the decoder

            #use greedy decode here for model output as well
            model_out = greedy_decode(model, encoder_input, encoder_mask, tokenizer_src, tokenizer_tgt, max_len, device)
            #compare this model output with what we expected with the label
            #the model output was the source_texts[], expected[], predicted[]
            #at the end of the loop we print them on the console
            source_text = batch['src_text'][0]
            target_text = batch['tgt_text'][0]
            #to get the text of the output of the model, need to use tokenizer to convert the tokens back into text
            #we use the target tokenizer bc it's the target language we're looking at for output
            model_out_text = tokenizer_tgt.decode(model_out.detach().cpu().numpy())

            #save into respective lists, this code is not necessary unless showing the output to tensorboard
            #source_texts.append(source_text)
            #expected.append(target_text)
            #predicted.append(model_out_text)

            #print to the console using print_msg, not usual print statement
            #reason: we're using in the training loop: tqdm to print the progress bar
            #tqdm suggests not to use usual print statement to console while the progress bar is running
            #so use print_msg to avoid interfering with the progress bar
            print_msg('-'*console_width)
            print_msg(f'SOURCE: {source_text}')
            print_msg(f'TARGET: {target_text}')
            print_msg(f'PREDICTED: {model_out_text}')

            #if we have processed the number of examples, then break
            if count == num_examples:
                break

    #tensorboard: to show the outputs
    #need another library to calculate metrics
    #TorchMetrics CharErrorRate, BLEU (for translation tasks), WordErrorRate
    #if writer: this code is in GitHub

def get_all_sentences(ds, lang):
    for item in ds:
        #recall the dataset is made of pairs of sentences: src, tgt. extract 1 language
        #this extracts only 1 language from the pair we want
        yield item['translation'][lang]


#method to build the tokenizer, config of model, dataset, language of tokenizer
#this code comes from huggingface tokenizer library for increase open source use
#building a tokenizer would be reinventing the wheel
def get_or_build_tokenizer(config, ds, lang):
    #file we'll save the tokenizer, placeholder for tokenizer_file: the path to the tokenizer file
    #this path is formatable: config['tokenizer_file'] = '../tokenizers/tokenizer_{0}.json'
    #given the lang param, it will create a tokenizer in yf language
    tokenizer_path = Path(config['tokenizer_file'].format(lang))
    #if tokenizer does not exist, we create it
    #unk_token: for unknown words, if the tokenizer sees a word that does not exist in its vocabulary
    #then replace that word with this word: UNK, and map it to the # that corresponds to 'UNK'
    if not Path.exists(tokenizer_path):
        tokenizer = Tokenizer(WordLevel(unk_token='[UNK]'))
        #pre_tokenizer: means we split word boundaries by Whitespace
        tokenizer.pre_tokenizer = Whitespace
        #build a wordlevel trainer to train tokenizer. w/4 special tokens.
        #PAD is used to train the transformer
        #min_frequency: for a word to appear in our vocabulary, it must appear at minimum=2
        trainer = WordLevelTrainer(special_tokens=["[UNK]", "[PAD]", "[SOS]", "[EOS]"], min_frequency=2)
        #train the tokenizer, get all sentences from our dataset
        tokenizer.train_from_iterator(get_all_sentences(ds, lang), trainer=trainer)
        tokenizer.save(str(tokenizer_path))
    else:
        tokenizer = Tokenizer.from_file(str(tokenizer_path))
    return tokenizer

#get dataset
#name the loaded dataset, ds_raw
def get_ds(config):
    #huggingface lets us dl datasets by dataset name, and the subset: eng-italian
    #abstract it by language_source and language_target
    #define the training & validation split
    ds_raw = load_dataset('opus_books', f'{config["lang_src"]}-{config["lang_tgt"]}', split='train')

    #build tokenizers
    tokenizer_src = get_or_build_tokenizer(config, ds_raw, config['lang_src'])
    tokenizer_tgt = get_or_build_tokenizer(config, ds_raw, config['lang_tgt'])

    #since we only have training split from huggingface, we can split it ourselves for training (90%) & validation (10%)
    train_ds_size = int(0.9 * len(ds_raw))
    val_ds_size = len(ds_raw) - train_ds_size
    #random_split is method from pytorch that splits dataset(ds_raw) based on input we give it: train_ds_size and val_ds_size
    train_ds_raw, val_ds_raw = random_split(ds_raw, [train_ds_size, val_ds_size])

#Create bilingual dataset so our model can access the tensors directly
#see file: dataset.py
#once dataset.py can create a dataset, we create 2 datasets
#dataset1: for training; dataset2: for validation
#then send to dataloader, then finally to the training loop
#The config will contain: src lang, tgt lang, Seq_Len

    train_ds = BilingualDataset(train_ds_raw, tokenizer_src, tokenizer_tgt, config['lang_src'], config['lang_tgt'], config['seq_len'])
#repeat for validation dataset
    val_ds = BilingualDataset(val_ds_raw, tokenizer_src, tokenizer_tgt, config['lang_src'], config['lang_tgt'], config['seq_len'])

#to choose max Seq_Len, we also want to watch what the max length of each sentence in src and tgt for each of the 2 splits we created here
#so if we choose very small Seq_Len, we will know

    max_len_src = 0
    max_len_tgt = 0

    for item in ds_raw:
        #load each sentence from each src and tgt language
        #which we convert into IDs using tokenizer, then check length
        #if max length is 180, then we can choose 200 to cover all possible sentences in the dataset
        #if max length is 500, then we can choose 510 to cover all possible sentences in the dataset
        #bc we also need to add SOS, EOS tokens to the sentences
        src_ids = tokenizer_src.encode(item['translation'][config['lang_src']]).ids
        tgt_ids = tokenizer_src.encode(item['translation'][config['lang_tgt']]).ids
        #src max length is max length of src and same for tgt
        max_len_src = max(max_len_src, len(src_ids))
        max_len_tgt = max(max_len_tgt, len(tgt_ids))

        print(f'Max length of source sentence: {max_len_src}')
        print(f'Max length of target sentence: {max_len_tgt}')

        #create dataloader
        #define batch size acc to our config
        train_dataloader = DataLoader(train_ds, batch_size=config['batch_size'], shuffle=True)
        #validation, use batch_size=1 to process each sentence 1 by one. This is where local 'short/long term memory is determined'
        val_dataloader = DataLoader(val_ds, batch_size=1, shuffle=True)

        #this method get_dataset, returns: dataloader of training, dataloader of validation, tokenizer of src language, tokenizer of tgt language

        return train_dataloader, val_dataloader, tokenizer_src, tokenizer_tgt

        #build the model, according to our config, src vocab size and tgt vocab size
def get_model(config, vocab_src_len, vocab_tgt_len):
    #build it based on src vocab size and tgt vocab size, the Seq_Len in config, the d_model size of the embedding
    model = build_transformer(vocab_src_len, vocab_tgt_len, config['seq_len'], config['seq_len'], config['d_model'] )
    return model

    #if the model is too big for your GPU to train on, then reduce # of heads or # of layers
    #this will impact performance of the model
    #this dataset is not so big/complex, it shouldn't be a big prob, since we're not building a huge dataset anyway

    #see file: dataset.py
    
    #now let's define the Configuration file
    #see file: config.py
