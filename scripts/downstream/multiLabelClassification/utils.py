
# def load_pretrained_model(model_name):
#     # Load model, the pretrained model will include a single linear classification layer on top for classification. 
#     if model_name=="roberta":
#         model = RobertaForSequenceClassification.from_pretrained("/mnt/localdata/geng/model/legalRoberta/", num_labels=NUM_LABELS)
#     if model_name=="legalBert":
#         model = BertForSequenceClassification.from_pretrained("nlpaueb/legal-bert-base-uncased", num_labels=NUM_LABELS)
#     if model_name=="bert_uncased":
#         model = BertForSequenceClassification.from_pretrained("bert-base-uncased",num_labels=NUM_LABELS)
#     if model_name=="bert_cased":
#         model = BertForSequenceClassification.from_pretrained("bert-base-cased",num_labels=NUM_LABELS)
#     if model_name=="bert_large":
#         model = BertForSequenceClassification.from_pretrained("bert-large-cased",num_labels=NUM_LABELS)
#     if model_name=="gpt2":
#         model = GPT2ForSequenceClassification.from_pretrained("gpt2",num_labels=NUM_LABELS)
#     return model

# def load_sequenceClassificationModel(model_name)
#     # Load model, the pretrained model will include a single linear classification layer on top for classification. 
#     if model_name=="roberta":
#         model = RobertaForSequenceClassification.from_pretrained("/mnt/localdata/geng/model/legalRoberta/", num_labels=NUM_LABELS)
#     if model_name=="legalBert":
#         model = BertForSequenceClassification.from_pretrained("nlpaueb/legal-bert-base-uncased", num_labels=NUM_LABELS)
#     if model_name=="bert_uncased":
#         model = BertForSequenceClassification.from_pretrained("bert-base-uncased",num_labels=NUM_LABELS)
#     if model_name=="bert_cased":
#         model = BertForSequenceClassification.from_pretrained("bert-base-cased",num_labels=NUM_LABELS)
#     if model_name=="bert_large":
#         model = BertForSequenceClassification.from_pretrained("bert-large-cased",num_labels=NUM_LABELS)
#     if model_name=="gpt2":
#         model = GPT2ForSequenceClassification.from_pretrained("gpt2",num_labels=NUM_LABELS)

#     return 