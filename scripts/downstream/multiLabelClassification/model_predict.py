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
from transformers import AutoTokenizer, AutoModel
import pickle
import json

#clf
from transformers import  RobertaForSequenceClassification
from transformers import  BertForSequenceClassification



if __name__ == '__main__':
    import argparse, sys

    parser = argparse.ArgumentParser()
    #首先是mandatory parameters
    parser.add_argument("model_name", help="legalBert vs roberta",choices=['legalBert','roberta'])# 必须以 python script.py mandatory_para_value....(如果没有这个参数会报错)

    #解析参数
    args = parser.parse_args()


    NUM_LABELS=4193
    # Select a batch size for training. For fine-tuning with XLNet, the authors recommend a batch size of 32, 48, or 128. We will use 32 here to avoid memory issues.
    batch_size = 32

    #max number of input tokens for one sentence
    max_length = 512 

    # set the expeiment model name
    model_name=args.model_name


    #cuda
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    n_gpu = torch.cuda.device_count()





    test_dataloader=torch.load('/mnt/localdata/geng/data/downstream/multiLabelClassification/{}/test_data_loader'.format(model_name))



    # Load model, the pretrained model will include a single linear classification layer on top for classification. 
    if model_name=="roberta":
        model = RobertaForSequenceClassification.from_pretrained("/mnt/localdata/geng/model/legalRoberta/", num_labels=NUM_LABELS)
    if model_name=="legalBert":
        model = BertForSequenceClassification.from_pretrained("nlpaueb/legal-bert-base-uncased", num_labels=NUM_LABELS)
    if model_name=="bert_uncased":
        model = BertForSequenceClassification.from_pretrained("bert-base-uncased",num_labels=NUM_LABELS)
    if model_name=="bert_cased":
        model = BertForSequenceClassification.from_pretrained("bert-base-cased",num_labels=NUM_LABELS)
    if model_name=="bert_large":
        model = BertForSequenceClassification.from_pretrained("bert-large-cased",num_labels=NUM_LABELS)
    if model_name=="gpt2":
        model = GPT2ForSequenceClassification.from_pretrained("gpt2",num_labels=NUM_LABELS)

    model.load_state_dict(torch.load( '/mnt/localdata/geng/model/lmtc_models/downstream/multiLabelClassification/{0}/clf_{0}'.format(model_name)))
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
            outs = parallel_model(b_input_ids, token_type_ids=None, attention_mask=b_input_mask)
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



    with open("/mnt/localdata/geng/model/lmtc_models/downstream/multiLabelClassification/{0}/prediction.pickle".format(model_name), "wb") as f:
        pickle.dump((pred_labels,true_labels), f)

