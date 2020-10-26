import spacy
from spacy.lang.en import English

with open ("../data/processed/Ensemble/Ensemble_v0.txt","r") as file:
    Ensemble = file.read()

# ran ge(40) needs more than 300G RAM
Ensemble_trunks=[Ensemble[n*2**28:(n+1)*2**28] for n in range(20)]

nlp = English()  # just the language with no model
sentencizer = nlp.create_pipe("sentencizer")
nlp.add_pipe(sentencizer)
nlp.max_length=2**30
# nlp.disable_pipes(["tagger", "parser", "ner"])
for i in range(len(Ensemble_trunks)):
    trunk=Ensemble_trunks[i]
    doc = nlp(trunk)
    with open("../data/processed/Ensemble/Ensemble_trunk{}.txt".format(i),"w") as file:
        for sent in doc.sents:
            file.write(sent.text+"\n")