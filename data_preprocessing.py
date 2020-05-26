import json
import os
import nltk
import string
from gensim.models import FastText
from gensim.models import KeyedVectors
import inflect
import time
import numpy as np
start = time.time()
current_dir = os.getcwd()

input_file = os.path.join(current_dir, 'train.json')
model = KeyedVectors.load_word2vec_format('wiki-news-300d-1M.vec')
print('model load time: ', time.time() - start)

p = inflect.engine()
with open(input_file, 'r') as f:
    word_relations = []
    for line in f:
        data_dic = json.loads(line)
        word_dict={}
        POS = nltk.pos_tag(data_dic['tokens'])
        for i in data_dic['nodes']:
            word = []
            for word_num in range(i[0][0], i[0][1]):
                word_vector = np.zeros(300)
                word.append(data_dic['tokens'][word_num])
                try:
                    new_word = float(data_dic['tokens'][word_num])
                    new_word = round(new_word)
                except:
                    new_word = data_dic['tokens'][word_num].lower()
                    new_word = new_word.replace(',','')
                    new_word = new_word.replace(r'\/','')
                    #print(new_word)

                try:
                    word_vector += model[str(new_word)]
                except:
                    try:
                        new_word = int(new_word)
                        word_vector += model[str(1000)]
                    except:
                        word_vector +=  np.zeros(300)

                #print('model load time: ', time.time() - start)
            word_vector /= (i[0][1] - i[0][0])

            for pos_num in range(i[0][0], i[0][1]):
                if POS[pos_num][1] not in string.punctuation:
                    pos_word = POS[pos_num][1]
                    break
            for key, value in i[1].items():
                word_node = key
            word_dict.update({i[0][0]:[word, key, i[0][0],pos_word, word_vector.tolist()]})

        for j in data_dic['edges']:
            temp_case = []
            temp_case.append(word_dict[j[0][0]])
            temp_case.append(word_dict[j[1][0]])
            for key, value in j[2].items():
                temp_case.append(key)
            word_relations.append(temp_case)

with open('train_processed.json', 'w') as f:
    json.dump({'Relation':word_relations}, f)
print(word_relations)
#print(nltk.pos_tag(data_dic['tokens']))
print('model load time: ', time.time() - start)
