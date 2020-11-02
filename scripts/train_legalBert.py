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

#clf
from transformers import  RobertaForSequenceClassification
from transformers import  BertForSequenceClassification


NUM_LABELS=4193
# Select a batch size for training. For fine-tuning with XLNet, the authors recommend a batch size of 32, 48, or 128. We will use 32 here to avoid memory issues.
batch_size = 32

#max number of input tokens for one sentence
max_length = 512 

# set the expeiment model name
model_name="legalBert"
#cuda
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
n_gpu = torch.cuda.device_count()

# import pretrained bert tokenizer
if model_name=="roberta":
    tokenizer = AutoTokenizer.from_pretrained("/mnt/localdata/geng/model/legalRoberta/", do_lower_case=True) 
if model_name=="legalBert":
    tokenizer = AutoTokenizer.from_pretrained("nlpaueb/legal-bert-base-uncased")

print("torch.cuda.is_available: ",torch.cuda.is_available())





# Load Model & Set Params

validation_dataloader=torch.load('/mnt/localdata/geng/data/downstream/multiLabelClassification/{}/validation_data_loader'.format(model_name))
train_dataloader=torch.load('/mnt/localdata/geng/data/downstream/multiLabelClassification/{}/train_data_loader'.format(model_name))




# Load model, the pretrained model will include a single linear classification layer on top for classification. 
if model_name=="roberta":
    model = RobertaForSequenceClassification.from_pretrained("/mnt/localdata/geng/model/legalRoberta/", num_labels=NUM_LABELS)
if model_name=="legalBert":
    model = BertForSequenceClassification.from_pretrained("nlpaueb/legal-bert-base-uncased", num_labels=NUM_LABELS)


    
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

# Store our loss and accuracy for plotting
train_loss_set = []

# Number of training epochs (authors recommend between 2 and 4)
epochs = 1

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

        # Forward pass for multilabel classification
        outputs = parallel_model(b_input_ids, token_type_ids=None, attention_mask=b_input_mask)
        logits = outputs[0]
        loss_func = BCEWithLogitsLoss() 
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

    print('F1 Validation Accuracy: ', val_f1_accuracy)
    print('Flat Validation Accuracy: ', val_flat_accuracy)


torch.save(model.state_dict(), '/mnt/localdata/geng/model/lmtc_models/downstream/multiLabelClassification/{0}/clf_{0}'.format(model_name))


