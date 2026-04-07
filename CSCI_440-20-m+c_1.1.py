#filename: CSCI_440-21-m+c_1.0.py

#I. Math


#II. Code

#the following code is for file: weights_1.0.py

#2:20 Validation Step
#weights file should be created at end of first epoch


#2:41
#now look at a model pretrained for a few hours so we can inference it and visualize the attention
#can try to copy the same code
#ran validation on this pretrained model, the model inferences num_examples=10. result is not bad, nearly overfit.
#this is the power of the transformer, only trained for a few hours (not days) and the results are really good.

#the following code is for file: train_1.0.py

#build the training loop
def train_model(config):
    #define which device to put all the tensors. Can be GPU cuda or CPU
    #check your computer for auto shut off, sleep, suspend battery settings
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device {device}')

    #ensure the weights folder is created
    Path(config['model_folder']).mkdir(parents=True, exist_ok=True)

    #load dataset
    train_dataloader, val_dataloader, tokenizer_src, tokenizer_tgt = get_ds(config)
    # to get vocab size, there's methods to get vocab size, and transfer the model to the GPU/CPU device
    model = get_model(config, tokenizer_src.get_vocab_size(), tokenizer_tgt.get_vocab_size().to(device))

    # start TensorBoard to visualize the loss graphics/charts
    writer = SummaryWriter(config['experiment_name'])

    #create optimizer: Adam
    optimizer = torch.optim.Adam(model.parameters(), lr=config['lr'], eps=1e-9)

    #we added config to resume training if model crashes in file: config.py (preload)
    #implemented here to restore state of the model and the state of the optimizer
    initial_epoch = 0
    global_step = 0
    if config['preload']:
        model_filename = get_weights_file_path(config, config['preload'])
        print(f'Preloading model {model_filename}')
        state = torch.load(model_filename)
        initial_epoch = state['epoch'] + 1
        optimizer.load_state_dict(state['optimizer_state_dict'])
        global_step = state['global_step']

        #the loss function we'll use: cross entropy loss
        #indicate ignore index, ignore padding, we don't want PAD token to contribute to the loss
        #also use label smoothing: allows model to be less confident about its decision
        #imagine the model is telling us to choose word #3 w/high probability
        #so with label smoothing, we take a little percentage of that probability and distribute it to the other tokens
        #so that the model becomes less sure of its choices, less overfitting, thus improving model accuracy
        #use label_smoothing=0.1, from every highest probability token, take 0.1% of score and give it to the others
        loss_fn = nn.CrossEntropyLoss(ignore_index=tokenizer_src.token_to_id('[PAD]'), label_smoothing=0.1).to(device)

        #build the training loop
        for epoch in range(initial_epoch, config['num_epochs']):
        #tell the model to train

            #batch iterator: for data loader, tqdm, will show nice progress bar
            batch_iterator = tqdm(train_dataloader, desc=f'Processing epoch {epoch:02d}')
            for batch in batch_iterator:
                model.train()

                #get the tensors for Encoder input
                encoder_input = batch['encoder_input'].to(device) #size is (Batch, Seq_Len)
                decoder_input = batch['decoder_input'].to(device) #size is (Batch, Seq_Len)
                encoder_mask = batch['encoder_mask'].to(device) #size is (Batch, 1, 1, Seq_Len)
                decoder_mask = batch['decoder_mask'].to(device) #size is (Batch, 1, Seq_Len, Seq_Len)
                #why are these last 2 masks different: bc in encoder, we're asking to hide PAD tokens
                #why are these last 2 masks different: bc in decoder, we're asking to hide all subsequent words

                #Run the tensors through the transformer
                #calculate output of Encoder
                encoder_output = model.encode(encoder_input, encoder_mask)
                #output of model.encode is (Batch, Seq_Len, d_model)
                decoder_output = model.decode(encoder_output, encoder_mask, decoder_input, decoder_mask)
                #output of model.decode is (Batch, Seq_Len, d_model), the same as encoder
                #need to map it back to the projection
                proj_output = model.project(decoder_output) # (Batch, Seq_Len, tgt_vocab_size)

                #now that we have the output of the model, we want to compare it to our label
                #extract label from the batch

                label = batch['label'].to(device) # put it on our device, (Batch, Seq_Len)
                #label is for each Batch and Seq_Len dimension, tells us what is the position in the vocabulary of that particular word
                #we want proj_output (Batch, Seq_Len, tgt_vocab_size) to be comparable to label (Batch, Seq_Len)

                #first compute loss
                #this transforms the proj_output (Batch, Seq_Len, tgt_vocab_size) to --> (Batch * Seq_Len, tgt_vocab_size) to compare it with batch['label'].to(device)
                #this is how the cross entropy wants the tensors to be (math requirement) and the label
                loss = loss_fn(proj_output.view(-1, tokenizer_tgt.get_vocab_size()), label.view(-1))

                #now that we've calculated the loss, we can update our progress bar with the loss we calculated
                batch_iterator.set_postfix({f"loss": f"{loss.item():6.3f}"})

                #log the loss on tensorboard
                writer.add_scalar('train loss', loss.item(), global_step)
                writer.flush()

                #backpropagate the loss
                loss.backward()

                #update the weights of the model
                optimizer.step()
                optimizer.zero_grad()

                #move the global step by 1
                #the global step is being used by tensor board to keep track of the loss
                global_step += 1

                #save the model at the end of every epoch
                #it's good to resume the training to save the state of the model and the state of the optimizer
                #bc the optimizer also keeps track of some statistics, 1 for each weight, to undersand how to move each weight independently
                #the optimizer dictionary is quite big, so if you want training to be resumable, you need to save it
                #otherwise the optimizer will start from zero and have to figure out from zero even if you start from a previous epoch how to move each weight

            #best practice: run validation after epoch, give it all the params needed to run the validation
            #lambda for printing messages via tqdm
            run_validation(model, val_dataloader, tokenizer_src, tokenizer_tgt, config['seq_len'], device, lambda msg: batch_iterator.write(msg), global_step, writer)
            #for PREDICTED: it will just output a bunch of commas during validation bc it's not training it at all
            #if we were training it, we would see PREDICTED values after a few epochs, and improve


            # Save the model at the end of every epoch
            model_filename = get_weights_file_path(config, f'{epoch:02d}')
            torch.save({
                'epoch': epoch,
                #model.state_dict() a dictionary object that maps each layer of a torch.nn.Module to its corresponding parameter tensor (weights and biases)
                #Contents: in includes all learnable params (weights, biases, from linear or convolutional layers) and registered buffers (e.g. running_mean, running_var from Batch Normalization)
                #Modularity: Because it's a standard py dictionary, it can easily be saved, updated, and altered, which is essential for saving and loading models.
                #model_state_dict() returns a reference to the parameters not a copy.
                #if you need to store the current state for later comparison (e.g. saving a 'best or golden' model), use: copy.deepcopy(model.state_dict())
                'model_state_dict': model.state_dict(),

                'optimizer_state_dict': optimizer.state_dict(),
                'global_step': global_step
            }, model_filename)

            #Comment this out for this class, but apply in real world: time.sleep(180)
    
    #build code to run this
    if __name__ == '__main__':
        warnings.filterwarnings('ignore')
        config = get_config()
        train_model(config)

#Lecture 19: stopped here

    #if this works, we expect this code should:
    #download the dataset
    #create the tokenizer, save it into its file
    #start training the model for 30 epochs (won't finish)

#2:20:42 Validation: want to visualize the output of the model while training
#want to see how the model evolves during training
#build Validation loop: to allow evaluation of the model, inference from this model
#check sample sentences from the model to see how accurate the translation is
