import json
import nltk
import string
from gensim.models import KeyedVectors
import inflect
import time
import numpy as np

p = inflect.engine()

def ReadFile(input_file, model):
    with open(input_file, 'r') as f:
        word_relations = []
        for line in f:
            data_dic = json.loads(line)
            word_dict={}
            POS = nltk.pos_tag(data_dic['tokens'])

            for node_index in data_dic['nodes']:
                word_vector = np.zeros(300)
                POS_vector = np.zeros(300)
                node_vector = np.zeros(300)

                #range(i[0][0], i[0][1]) means the word that we interest
                for index in range(node_index[0][0], node_index[0][1]):
                    #delete some punctuation
                    word  = data_dic['tokens'][index]
                    word =word.lower()
                    word = word.replace(r'\/','')
                    word = word.replace(r',','')

                    #the word which is not in dictionary
                    try:
                        word_vector += model[str(word)]
                    except:
                        word_vector +=  np.zeros(300)

                    #the POS which is not in dictionary
                    try:
                        POS_vector += model[POS[index][1]]
                    except:
                        word_vector +=  np.zeros(300)

                word_vector /= (node_index[0][1]- node_index[0][0])
                POS_vector /= (node_index[0][1]- node_index[0][0])
                
                for key, value in node_index[1].items():
                    #the node which is not in dictionary
                    try:
                        node_vector += model[key]
                    except:
                        node_vector = np.zeros(300)

                word_dict.update({node_index[0][0]:[word_vector.tolist(), node_vector.tolist(), node_index[0][0], POS_vector.tolist()]})
            #range(i[0][0], i[0][1]) means the relation that we interest
            for j in data_dic['edges']:
                for key, value in j[2].items():
                    word_relations.append([word_dict[j[0][0]], word_dict[j[1][0]], key])

    with open('Process' + input_file, 'w') as f:
        json.dump({'Relation':word_relations}, f)

    



if __name__ == '__main__':
    start = time.time()
    model = KeyedVectors.load_word2vec_format('wiki-news-300d-1M.vec')
    print('model load time: ', time.time() - start)
    TrainData = 'train.json'
    TestData = 'test.json'
    ReadFile(TestData, model)
