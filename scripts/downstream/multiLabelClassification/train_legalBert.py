import pandas as pd
import numpy as np
import tensorflow as tf
import torch
from torch.nn import BCEWithLogitsLoss, BCELoss
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics import classification_report, confusion_matrix, multilabel_confusion_matrix, f1_score, accuracy_score
import pickle
from transformers import *
from tqdm import tqdm, trange
from ast import literal_eval
from transformers import AutoTokenizer, AutoModel,AutoModelForSequenceClassification
import os
#clf
from transformers import  GPT2ForSequenceClassification
import json
import pdb


def create_if_not_exists(folder_fn):
    if not os.path.exists(folder_fn):
        os.mkdir(folder_fn)


def create_train_val_loaders(src_fn, val_loader_fn,train_loader_fn,tokenizer,batch_size,max_length):
    # df = pd.read_csv('/mnt/localdata/geng/data/downstream/multiLabelClassification/train.csv',index_col=0)
    print("###################### start create train val dataloaders ########################")
    df = pd.read_csv(src_fn)

    cols = df.columns
    label_cols = list(cols[1:])
    # print('Label columns: ', label_cols)
    # df = df.sample(frac=1).reset_index(drop=True) #shuffle rows 会不会是因为这里？
    df['one_hot_labels'] = list(df[label_cols].values)
    labels = list(df.one_hot_labels.values)
    # pdb.set_trace()
    comments = list(df["text"].values)
    encodings = tokenizer(comments,max_length=max_length,truncation=True,pad_to_max_length=True,return_token_type_ids=True) 

    input_ids = encodings['input_ids'] # tokenized and encoded sentences
    token_type_ids = encodings['token_type_ids'] # token type ids
    # token_type_ids=[tokenizer.create_token_type_ids_from_sequences(input_id) for input_id in input_ids]
    # TODO, not sure which to use
    attention_masks = encodings['attention_mask'] # attention masks
    # Use train_test_split to split our data into train and validation sets

    train_inputs, validation_inputs, train_labels, validation_labels, train_token_types, validation_token_types, train_masks, validation_masks = train_test_split(input_ids, labels, token_type_ids,attention_masks,
                                                            random_state=2020, test_size=0.10)


    # Convert all of our data into torch tensors, the required datatype for our model
    train_inputs = torch.tensor(train_inputs)
    train_labels = torch.tensor(train_labels)
    train_masks = torch.tensor(train_masks)
    train_token_types = torch.tensor(train_token_types)

    validation_inputs = torch.tensor(validation_inputs)
    validation_labels = torch.tensor(validation_labels)
    validation_masks = torch.tensor(validation_masks)
    validation_token_types = torch.tensor(validation_token_types)



    # Create an iterator of our data with torch DataLoader. This helps save on memory during training because, unlike a for loop, 
    # with an iterator the entire dataset does not need to be loaded into memory

    train_data = TensorDataset(train_inputs, train_masks, train_labels, train_token_types)
    train_sampler = RandomSampler(train_data)
    train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=batch_size)

    validation_data = TensorDataset(validation_inputs, validation_masks, validation_labels, validation_token_types)
    validation_sampler = SequentialSampler(validation_data)
    validation_dataloader = DataLoader(validation_data, sampler=validation_sampler, batch_size=batch_size)

    torch.save(validation_dataloader,val_loader_fn)
    torch.save(train_dataloader,train_loader_fn)
    print("###################### finished to create train val dataloaders ########################")
    return validation_dataloader,train_dataloader



def create_test_loader(src_fn, test_loader_fn,tokenizer,batch_size,max_length):
    df = pd.read_csv(src_fn)

    cols = df.columns
    label_cols = list(cols[1:])
    # print('Label columns: ', label_cols)

    # df = df.sample(frac=1).reset_index(drop=True) #shuffle rows
    df['one_hot_labels'] = list(df[label_cols].values)
    labels = list(df.one_hot_labels.values)
    comments = list(df["text"].values)
    encodings = tokenizer(comments,max_length=max_length,truncation=True,pad_to_max_length=True,return_token_type_ids=True) 

    input_ids = encodings['input_ids'] # tokenized and encoded sentences
    token_type_ids = encodings['token_type_ids'] # token type ids
    # token_type_ids=[tokenizer.create_token_type_ids_from_sequences(input_id) for input_id in input_ids]
    # TODO, not sure which to use
    attention_masks = encodings['attention_mask'] # attention masks

    # Convert all of our data into torch tensors, the required datatype for our model
    test_inputs = torch.tensor(input_ids)
    test_labels = torch.tensor(labels)
    test_masks = torch.tensor(attention_masks)
    test_token_types = torch.tensor(token_type_ids)



    # Create an iterator of our data with torch DataLoader. This helps save on memory during testing because, unlike a for loop, 
    # with an iterator the entire dataset does not need to be loaded into memory

    test_data = TensorDataset(test_inputs, test_masks, test_labels, test_token_types)
    test_sampler = RandomSampler(test_data)
    test_dataloader = DataLoader(test_data, sampler=test_sampler, batch_size=batch_size)

    torch.save(test_dataloader,test_loader_fn)



if __name__ == '__main__':

    import argparse, sys

    parser = argparse.ArgumentParser()

    # given a list of choice 
    parser.add_argument("--task",default="multiLabelClassification",choices=["multiLabelClassification","twitter","singleLabelClassification"])
    parser.add_argument("--model_name", help="legalBert vs roberta",choices=["legalBert","legalRoberta","bert_uncased","bert_cased","bert_large","gpt2","roberta"])
    parser.add_argument("--build_new_dataloaders",action='store_true')
    parser.add_argument("-bs","--batch_size",type=int,default=None)
    parser.add_argument("--cpu",action='store_true')


        

    #解析参数
    args = parser.parse_args()

    task=args.task
    if task=="twitter":
        with open("config_twitter.json", "r") as read_file:
            config = json.load(read_file)
    elif task=="singleLabelClassification":
        with open("config_singleLabelClassification.json", "r") as read_file:
            config = json.load(read_file)
    elif task=="multiLabelClassification":
        with open("config.json", "r") as read_file:
            config = json.load(read_file)



    cpu=args.cpu

    #cuda
    if cpu:
        device = torch.device("cpu")
        print("using multi cpu mode")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        n_gpu = torch.cuda.device_count()
        print("torch.cuda.is_available: ",torch.cuda.is_available())
        print("torch.cuda.device_count:",n_gpu)


    # set the expeiment model name
    model_name=args.model_name
    build_new_dataloaders=args.build_new_dataloaders

    NUM_LABELS=config["task"]["NUM_LABELS"]

    if model_name=='gpt':
        max_length=config["task"]["max_length_gpt"]

    else:
        max_length=config["task"]["max_length_bert"]

    batch_size=config["task"]['batch_size']
    if args.batch_size:
        batch_size=args.batch_size





    # import pretrained bert tokenizer
    tokenizer = AutoTokenizer.from_pretrained(config['model'][model_name],use_fast=True)

    # pdb.set_trace()
    tokenizer.pad_token = "[PAD]"



    model_folder_fn='/mnt/localdata/geng/model/downstream/{task}/{model_name}'.format(model_name=model_name,task=task)
    data_folder_fn='/mnt/localdata/geng/data/downstream/{task}/{model_name}'.format(model_name=model_name,task=task)
    create_if_not_exists(data_folder_fn)
    create_if_not_exists(model_folder_fn)

    # Load Model & Set Params

    val_loader_fn='/mnt/localdata/geng/data/downstream/{task}/{model_name}/validation_data_loader_bs{batch_size}'.format(model_name=model_name,task=task,batch_size=batch_size)
    train_loader_fn='/mnt/localdata/geng/data/downstream/{task}/{model_name}/train_data_loader_bs{batch_size}'.format(model_name=model_name,task=task,batch_size=batch_size)
    test_loader_fn='/mnt/localdata/geng/data/downstream/{task}/{model_name}/test_data_loader_bs{batch_size}'.format(model_name=model_name,task=task,batch_size=batch_size)
    
    if os.path.exists(val_loader_fn) and (not build_new_dataloaders):
        print("train val dataloaders alreay exist, load them now.")
        validation_dataloader=torch.load(val_loader_fn)
        train_dataloader=torch.load(train_loader_fn)
    else:
        validation_dataloader,train_dataloader=create_train_val_loaders(config['dataset']['train'],val_loader_fn, train_loader_fn,tokenizer,batch_size,max_length)

    if os.path.exists(test_loader_fn) and (not build_new_dataloaders):
        pass
    else:
        create_test_loader(config['dataset']['test'],test_loader_fn,tokenizer,batch_size,max_length)



    #  import pretrained model 
    if model_name=="gpt2":
        model=GPT2ForSequenceClassification.from_pretrained("gpt2", num_labels=NUM_LABELS)
    else:
        model=AutoModelForSequenceClassification.from_pretrained(config['model'][model_name], num_labels=NUM_LABELS)
    
    if cpu:
        # Distributor = torch.nn.parallel.DistributedDataParallelCPU
        # import torch.distributed as dist
        # rank=1
        # world_size=12
        # dist.init_process_group("gloo", world_size=world_size,rank=-1, store= None)
        # parallel_model = Distributor(model)
        parallel_model=model
    else:
        parallel_model = torch.nn.DataParallel(model) # Encapsulate the model
        parallel_model.cuda()

    # setting custom optimization parameters. You may implement a scheduler here as well.
    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'gamma', 'beta']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
         'weight_decay_rate': 0.01},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)],
         'weight_decay_rate': 0.0}
    ]

    optimizer = AdamW(optimizer_grouped_parameters,lr=2e-5,correct_bias=True)
    # optimizer = AdamW(model.parameters(),lr=2e-5)  # Default optimization


    #Train
    print("start the training process ") 

    # Store our loss and accuracy for plotting
    train_loss_set = []

    # Number of training epochs (authors recommend between 2 and 4)
    epochs = config["task"]["epoch"]

    # trange is a tqdm wrapper around the normal python range
    for epoch__ in trange(epochs, desc="Epoch"):

        # Training

        # Set our model to training mode (as opposed to evaluation mode)
        parallel_model.train()

        # Tracking variables
        tr_loss = 0 #running loss
        nb_tr_examples, nb_tr_steps = 0, 0



        # Train the data for one epoch
        for step, batch in enumerate(train_dataloader):
            # Add batch to GPU
            batch = tuple(t.to(device) for t in batch)
            # Unpack the inputs from our dataloader
            b_input_ids, b_input_mask, b_labels, b_token_types = batch
            # Clear out the gradients (by default they accumulate)
            optimizer.zero_grad()

            # # Forward pass for multiclass classification
            # outputs = model(b_input_ids, token_type_ids=None, attention_mask=b_input_mask, labels=b_labels)
            # loss = outputs[0]
            # logits = outputs[1]
            # pdb.set_trace()
            # Forward pass for multilabel classification
            outputs = parallel_model(b_input_ids, token_type_ids=b_token_types, attention_mask=b_input_mask)
            logits = outputs[0]
            loss_func = BCEWithLogitsLoss() 
            # pdb.set_trace()
            loss = loss_func(logits.view(-1,NUM_LABELS),b_labels.type_as(logits).view(-1,NUM_LABELS)) #convert labels to float for calculation
            # loss_func = BCELoss() 
            # loss = loss_func(torch.sigmoid(logits.view(-1,NUM_LABELS)),b_labels.type_as(logits).view(-1,NUM_LABELS)) #convert labels to float for calculation
            train_loss_set.append(loss.item())    

            # Backward pass
            loss.mean().backward()
            # Update parameters and take a step using the computed gradient
            optimizer.step()
            # scheduler.step()
            # Update tracking variables
            tr_loss += loss.item()
            nb_tr_examples += b_input_ids.size(0)
            nb_tr_steps += 1

            print("Train loss: {}".format(tr_loss/nb_tr_steps))

        ###############################################################################

        # Validation

        # Put model in evaluation mode to evaluate loss on the validation set
        parallel_model.eval()

        # Variables to gather full output
        logit_preds,true_labels,pred_labels,tokenized_texts = [],[],[],[]

        # Predict
        for i, batch in enumerate(validation_dataloader):
            batch = tuple(t.to(device) for t in batch)
            # Unpack the inputs from our dataloader
            b_input_ids, b_input_mask, b_labels, b_token_types = batch
            with torch.no_grad():
                # Forward pass
                outs = parallel_model(b_input_ids, token_type_ids=b_token_types, attention_mask=b_input_mask)
                b_logit_pred = outs[0]
                pred_label = torch.sigmoid(b_logit_pred)

                b_logit_pred = b_logit_pred.detach().cpu().numpy()
                pred_label = pred_label.to('cpu').numpy()
                b_labels = b_labels.to('cpu').numpy()

            tokenized_texts.append(b_input_ids)
            logit_preds.append(b_logit_pred)
            true_labels.append(b_labels)
            pred_labels.append(pred_label)

        # Flatten outputs
        pred_labels = [item for sublist in pred_labels for item in sublist]
        true_labels = [item for sublist in true_labels for item in sublist]

        # Calculate Accuracy
        threshold = 0.50
        pred_bools = [pl>threshold for pl in pred_labels]
        true_bools = [tl==1 for tl in true_labels]
        val_f1_accuracy = f1_score(true_bools,pred_bools,average='micro')*100
        val_flat_accuracy = accuracy_score(true_bools, pred_bools)*100

        print('F1 Validation Accuracy: ', val_f1_accuracy)
        print('Flat Validation Accuracy: ', val_flat_accuracy)


    torch.save(model.state_dict(), '/mnt/localdata/geng/model/downstream/{task}/{model_name}/clf_{model_name}'.format(model_name=model_name,task=task))

    
