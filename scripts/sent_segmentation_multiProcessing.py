import multiprocessing
import spacy
from spacy.lang.en import English
import pickle

def worker(i, return_dict):
    """worker function"""
    with open("../data/processed/Ensemble_splits/Ensemble_split{}.txt".format(i),"r") as file:
        trunk=file.read()
    return_dict[i] =[str(sent)+"\n" for sent in list(nlp(trunk))]

    with open("../data/processed/Ensemble_splits/Ensemble_split{}.txt".format(i+5),"r") as file:
        trunk=file.read()
    return_dict[i] =[str(sent)+"\n" for sent in list(nlp(trunk))]

    with open("../data/processed/Ensemble_splits/Ensemble_split{}.txt".format(i+10),"r") as file:
        trunk=file.read()
    return_dict[i] =[str(sent)+"\n" for sent in list(nlp(trunk))]

    with open("../data/processed/Ensemble_splits/Ensemble_split{}.txt".format(i+15),"r") as file:
        trunk=file.read()
    return_dict[i] =[str(sent)+"\n" for sent in list(nlp(trunk))]

nlp = English()  # just the language with no model
sentencizer = nlp.create_pipe("sentencizer")
nlp.add_pipe(sentencizer)
nlp.max_length=2**30

manager = multiprocessing.Manager()
return_dict = manager.dict()
jobs = []

#6 split+6 process, 占用200G;单线程每1/20 split需要6分钟，6*20/60=2hour, 2hour/6=20mins
for i in range(5):
    p = multiprocessing.Process(target=worker, args=(i, return_dict))
    jobs.append(p)
    p.start()

for proc in jobs:
    proc.join()

with open("sent_segmentation.pkl", "wb") as f:
    pickle.dump(return_dict, f)