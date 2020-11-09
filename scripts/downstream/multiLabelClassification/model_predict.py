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
import pickle
import json
from utils import *
import pdb

#clf
from transformers import  GPT2ForSequenceClassification



if __name__ == '__main__':
    import argparse, sys

    parser = argparse.ArgumentParser()
    #首先是mandatory parameters
    parser.add_argument("--task",default="multiLabelClassification",choices=["multiLabelClassification","twitter"])
    parser.add_argument("--model_name", help="legalBert vs roberta",choices=["legalBert","legalRoberta","bert_uncased","bert_cased","bert_large","gpt2","roberta"])
    parser.add_argument("--cpu",action='store_true')
    parser.add_argument("-bs","--batch_size",type=int,default=None)
    #解析参数
    args = parser.parse_args()


    task=args.task

    if task=="twitter":
        with open("config_twitter.json", "r") as read_file:
            config = json.load(read_file)
    elif task=="multiLabelClassification":
        with open("config.json", "r") as read_file:
            config = json.load(read_file)


    NUM_LABELS=config["task"]["NUM_LABELS"]
    # Select a batch size for training. For fine-tuning with XLNet, the authors recommend a batch size of 32, 48, or 128. We will use 32 here to avoid memory issues.
    batch_size=config["task"]['batch_size']
    if args.batch_size:
        batch_size=args.batch_size

    # set the expeiment model name
    model_name=args.model_name
    cpu=args.cpu



    #max number of input tokens for one sentence
    if model_name=='gpt':
        max_length=config["task"]["max_length_gpt"]
    else:
        max_length=config["task"]["max_length_bert"]





    #cuda
    if cpu:
        device = torch.device("cpu")
        print("using multi cpu mode")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        n_gpu = torch.cuda.device_count()
        print("torch.cuda.is_available: ",torch.cuda.is_available())
        print("torch.cuda.device_count:",n_gpu)



    test_loader_fn='/mnt/localdata/geng/data/downstream/{task}/{model_name}/test_data_loader_bs{batch_size}'.format(model_name=model_name,batch_size=batch_size,task=task)
    if cpu:
        test_dataloader=torch.load(test_loader_fn,map_location=torch.device('cpu'))
    else:
        test_dataloader=torch.load(test_loader_fn)


    #  import pretrained model 
    if model_name=="gpt2":
        model=GPT2ForSequenceClassification.from_pretrained("gpt2", num_labels=NUM_LABELS)
    else:
        model=AutoModelForSequenceClassification.from_pretrained(config['model'][model_name], num_labels=NUM_LABELS)

    model_fn='/mnt/localdata/geng/model/downstream/{task}/{model_name}/clf_{model_name}'.format(model_name=model_name,task=task)
    if cpu:
        clf_model=torch.load(model_fn,map_location=torch.device('cpu'))
    else:
        clf_model=torch.load(model_fn)

    model.load_state_dict(clf_model)
    

    if cpu:
        parallel_model=model
    else:
        parallel_model = torch.nn.DataParallel(model) # Encapsulate the model
        parallel_model.cuda()




    # Put model in evaluation mode to evaluate loss on the test set
    parallel_model.eval()

    # Variables to gather full output
    logit_preds,true_labels,pred_labels,tokenized_texts = [],[],[],[]

    # Predict
    for i, batch in enumerate(test_dataloader):
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

    print('F1 test Accuracy: ', val_f1_accuracy)
    print('Flat test Accuracy: ', val_flat_accuracy)



    with open("/mnt/localdata/geng/model/downstream/{task}/{model_name}/prediction.pickle".format(model_name=model_name,task=task), "wb") as f:
        pickle.dump((pred_labels,true_labels), f)

