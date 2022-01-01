from keras.models import load_model
import numpy as np
from keras_contrib.layers import CRF
from keras_contrib.losses import crf_loss
from keras_contrib.metrics import crf_viterbi_accuracy
from transformers import AutoTokenizer
import string 
from spacy import displacy

save_model_address = 'b'
tokenizer = AutoTokenizer.from_pretrained(save_model_address, do_lower_case=True)

custom_objects={'CRF': CRF,'crf_loss': crf_loss,'crf_viterbi_accuracy':crf_viterbi_accuracy}
model = load_model('l/ner-bi-lstm-td-model-0.99.hdf5',custom_objects = custom_objects)

tag2idx = {'B-CERTIFICATION': 14,
            'B-FRAMEWORK': 3,
            'B-KNOWLEDGE': 18,
            'B-PLANGUAGE': 11,
            'B-PLATFORM': 2,
            'B-SOFTSKILL': 13,
            'B-TECHNIQUE': 8,
            'B-TOOL': 1,
            'I-CERTIFICATION': 4,
            'I-FRAMEWORK': 17,
            'I-KNOWLEDGE': 12,
            'I-PLANGUAGE': 7,
            'I-PLATFORM': 0,
            'I-SOFTSKILL': 19,
            'I-TECHNIQUE': 9,
            'I-TOOL': 10,
            'O': 5,
            'X': 6,
            '[CLS]': 15,
            '[SEP]': 16}
 # Mapping index to name
tag2name={tag2idx[key] : key for key in tag2idx.keys()}

def pred2label(pred):
    out_i = []
    for p in pred:
        p_i = np.argmax(p)
        #print(p_i)
        out_i.append(tag2name[p_i]) 
    
    return out_i

txt = '''You can start creating your own data science projects and collaborating with other data scientists using IBM Watson Studio. When you sign up, you will receive free access to Watson Studio. Start now and take advantage of this platform and learn the basics of programming, machine learning, and data visualization with this introductory course.'''
# Make text token into id
input=tokenizer(txt)
ids = input['input_ids'] 
len_ip=len(ids)
input_ids = [ids + [0] * (256 - len(ids))]
input_ids=np.array(input_ids)
input_ids_dims = input_ids.reshape(1, -1)
#print(input_ids_dims)

res=model.predict(input_ids_dims,verbose=1)
#print(res)

pred_labels = pred2label(res[0][0:len_ip])
#print(pred_labels)

ner_result=list(zip(tokenizer.convert_ids_to_tokens(ids),pred_labels))
#print(ner_result)

# List with the result
collapsed_result = []

# Buffer for tokens belonging to the most recent entity
current_entity_tokens = []
current_entity = None
current_start=0
current_end=0
count=0
# Iterate over the tagged tokens
for token, tag in ner_result:
    #print(tag)
    if token in string.punctuation:
        count+= 1
        continue
    if token in ["[CLS]","[SEP]"]:
        continue
    if tag in ["O"]:
        count=count + len(token) + 1
        continue
    # If an enitity span starts ...
    if tag.startswith("B-"):
                # ... if we have a previous entity in the buffer, store it in the result list
        if current_entity is not None:
            #print(current_start)
            #print(current_end)
            span = {'start':current_start,'end': current_end, 'label':current_entity}
            collapsed_result.append(span)
        
        current_start= count
        current_end= count + len(token)
        #print('B-TAG: ', current_start)
        #print('B-TAG: ', current_end)
        current_entity = tag[2:]
        # The new entity has so far only one token
        current_entity_tokens = [token]
    # If the entity continues ...
    elif tag == "I-" + current_entity:
        # Just add the token buffer
        current_entity_tokens.append(token)
        current_end = current_end + len(token) + 1
        #print('I-TAG: ',current_end)
    else:
        raise ValueError("Invalid tag order.")
    
    count= count + len(token) + 1
    

# The last entity is still in the buffer, so add it to the result
# ... but only if there were some entity at all
if current_entity is not None:
    span = {'start':current_start,'end': current_end, 'label':current_entity}
    collapsed_result.append(span)
print(collapsed_result)

doc = {'ents':collapsed_result,'text': txt}
print(doc)

displacy.serve(doc, style='ent', manual=True)