import pandas as pd
import numpy as np
import os
from pprint import pprint
import spacy
from spacy import displacy
#from transformers import BertTokenizer, BertForTokenClassification
from transformers import pipeline
from transformers import AutoTokenizer, AutoModelForTokenClassification

save_model_address = 'b'
#save_model = BertForTokenClassification.from_pretrained(save_model_address, num_labels=20)
#tokenizer = BertTokenizer.from_pretrained(save_model_address,do_lower_case=True)

save_model = AutoModelForTokenClassification.from_pretrained(save_model_address)
tokenizer = AutoTokenizer.from_pretrained(save_model_address, do_lower_case=True)

nlp = pipeline("ner", model=save_model, tokenizer=tokenizer, aggregation_strategy='simple',ignore_labels =['X','O'])

sentences = '''You can start creating your own data science projects and collaborating with other data scientists using IBM Watson Studio. When you sign up, you will receive free access to Watson Studio. Start now and take advantage of this platform and learn the basics of programming, machine learning, and data visualization with this introductory course.'''
results = nlp(sentences)
pprint(results)

ents = []
for ele in results: # add character indexes
    span = {'start':ele["start"],'end': ele["end"], 'label':ele["entity_group"]}
    ents.append(span)
doc = {'ents':ents,'text': sentences}
pprint(doc)

displacy.serve(doc, style='ent', manual=True)
