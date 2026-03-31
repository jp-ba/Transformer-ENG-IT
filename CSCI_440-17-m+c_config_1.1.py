#filename: CSCI_440-17-m+c_1.0.py

#I. Math


#II. Code
#The following code is for the file: config_1.0.py 
#define 2 methods: get config, get path to where we'll save weights of the model
from pathlib import Path

def get_config():
    return {
        "batch_size": 8,                #we choose 8, can choose bigger if your computer allows it
        "num_epochs": 20,               #number of epochs to train, 20 is enough
        "lr": 10**-4,                   #lr: learning rate. 10^-4, common to start w/high lr, then reduce gradually w/every epoch, we won't for coding simplicity
        "seq_len": 350,                 #for en-it, 350 is sufficient
        "d_model": 512,                 #default 512
        "lang_src": "en",               #language source: english
        "lang_tgt": "it",               #language target: italian
        "model_folder": "weights",      #where we save the weights
        "model_filename": "tmodel_",    #transformer
        "preload": None,                #preload the model in case we want to restart training after it crashes
        "tokenizer_file": "tokenizer_{0}.json",  #the tokenizer file, "en" if english, or "it" if italian
        "experiment_name": "runs/tmodel"                       #experiment name for tensor board to save the losses while training
    }

#define method to save the path where we save the model weights
def get_weights_file_path(config, epoch: str):
    model_folder = config['model_folder']
    model_basename = config['model_basename']
    model_filename = f"{model_basename}{epoch}.pt"
    return str(Path('.') / model_folder / model_filename)

