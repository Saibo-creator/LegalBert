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
from transformers import  RobertaForSequenceClassification


class MultiLabelClassifier(object):
    """docstring for MultiLabelClassifier"""
    def __init__(self, NUM_LABELS,BASE_MODEL):
        super(MultiLabelClassifier, self).__init__()
        self.NUM_LABELS=NUM_LABELS
        self.model
        self.data_loader_builder
        self.tokenizer
        self.BASE_MODEL=BASE_MODEL #"/mnt/localdata/geng/model/legalRoberta/"

    def load_tokenizer():
        self.tokenizer=tokenizer = RobertaTokenizer.from_pretrained(self.BASE_MODEL, do_lower_case=True) # tokenizer

    def load_base_model():
        self.model=RobertaForSequenceClassification.from_pretrained(self.BASE_MODEL, num_labels=NUM_LABELS)


class Trainer(object):
    """docstring for Trainer"""
    def __init__(self, model):
        super(Trainer, self).__init__()
        self.model = model
        


class Data_loader_builder(object):
    """docstring for ClassName"""
    def __init__(self, data_dir,train_fn,test_fn,eval_fn,max_length):
        super(ClassName, self).__init__()
        self.DATA_DIR = data_dir
        self.TRAIN_FN = train_fn
        self.TEST_FN = test_fn
        self.EVAL_FN = eval_fn
        self.MAX_LENGTH = max_length

    def load(self,path):
        dataloader=torch.load(path)
        return dataloader

    def save(self,dataloader,path):
        torch.save(dataloader,path)


    def build_train_val_data_loaders(partition,batch_size):
        if partition=="train":
            fn=self.TRAIN_FN
        elif partition=="test":
            fn=self.TEST_FN
        elif partition=="eval":
            fn=self.EVAL_FN

        df = pd.read_csv(fn,index_col=0)
        cols = df.columns
        label_cols = list(cols[2:])
        df['one_hot_labels'] = list(df[label_cols].values)
        labels = list(df.one_hot_labels.values)
        comments = list(df["header+recital"].values)

        encodings = self.tokenizer.batch_encode_plus(comments,max_length=self.MAX_LENGTH,truncation=True, pad_to_max_length=True) # tokenizer's encoding method
        print('tokenizer outputs: ', encodings.keys())

        input_ids = encodings['input_ids'] # tokenized and encoded sentences

        try:
            token_type_ids = encodings['token_type_ids'] # token type ids
        except Exception as e:# TODO
            token_type_ids=[tokenizer.create_token_type_ids_from_sequences(input_id) for input_id in input_ids]
        
        attention_masks = encodings['attention_mask'] # attention masks

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

        return train_dataloader,validation_dataloader

    def 