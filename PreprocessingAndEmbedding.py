import json
import nltk
import string
import gensim.downloader
import inflect
import time
import numpy as np

#t-sne reduction
from sklearn.manifold import TSNE


p = inflect.engine()

#nltk pos list
PosList = ["$", "''", "(", ")", ",", "--", ".", ":","CC", "CD", "DT", "EX", "FW", "IN", "JJ", "JJR", "JJS", "LS", "MD", 
           "NN", "NNP", "NMPS", "NNS", "PDT", "POS", "PRP", "PRP$", "RB", "RBR", "RBS", "RP", "SYM", "TO", "UH", "VB", 
           "VBD", "VBG", "VBN","VBZ", "WDT", "WP", "WP$", "WRB"]

NodeList = ["value", "agent", "condition", "theme", "theme_mod", "quant_mod", "co_quant", "null", "location", 
            "whole", "source", "reference_time", "quant", "manner", "time", "cause", "+", "-"]

NerList = ['B-FACILITY', 'B-GPE', 'B-GSP', 'B-LOCATION', 'B-ORGANIZATION', 'B-PERSON', 
            'I-FACILITY', 'I-GPE', 'I-GSP', 'I-LOCATION', 'I-ORGANIZATION', 'I-PERSON',
            'O']


PosTagtoNum = {tag: i for i, tag in enumerate(PosList)} 
NodeTagtoNum = {tag: i for i, tag in enumerate(NodeList)} 
NerTagtoNum = {tag: i for i, tag in enumerate(NerList)} 

wordEmbeddingSize = 100

def ReadFile(input_file, model):
    with open(input_file, 'r') as f:
        word_relations = []
        for line in f:
            data_dic = json.loads(line)
            word_dict={}
            pos = nltk.pos_tag(data_dic['tokens'])
            Chunk = nltk.ne_chunk(pos)
            PosNerList = nltk.chunk.tree2conlltags(Chunk)

            for node_index in data_dic['nodes']:
                word_vector = np.zeros(wordEmbeddingSize)
                pos_vector = np.zeros(len(PosList))
                ner_vector = np.zeros(len(NerList))

                #range(i[0][0], i[0][1]) means the word that we interest
                for index in range(node_index[0][0], node_index[0][1]):
                    #delete some punctuation
                    word  = data_dic['tokens'][index]
                    word =word.lower()
                    word = word.replace(r'\/','')
                    word = word.replace(r',','')

                    #word2vector 
                    try:
                        word_vector += model[str(word)]
                    except:
                        word_vector +=  np.zeros(wordEmbeddingSize)


                    #POS tag for one hot encoding
                    pos_feature =  np.zeros(len(PosList))
                    posTagNumber = PosTagtoNum.get(PosNerList[index][1], -1)
                    if posTagNumber != -1:
                        pos_feature[posTagNumber] = 1.0
                    
                    pos_vector += pos_feature


                    #NER tag for one hot encoding
                    ner_feature =  np.zeros(len(NerList))

                    #non-special name -> O
                    nerTagNumber = NerTagtoNum.get(PosNerList[index][2], -1)
                    
                    ner_feature[nerTagNumber] = 1.0                    
                    ner_vector += ner_feature

                word_vector /= (node_index[0][1]- node_index[0][0])
                pos_vector /= (node_index[0][1]- node_index[0][0])
                ner_vector /= (node_index[0][1]- node_index[0][0])

                for key, value in node_index[1].items():
                    #node for one hot encoding
                    node_feature =  np.zeros(len(NodeList))
                    nodeTagNumber = NodeTagtoNum.get(key, -1)
                    if nodeTagNumber != -1:
                        node_feature[nodeTagNumber] = 1.0

                word_dict.update({node_index[0][0]:[word_vector.tolist(), node_feature.tolist(), node_index[0][0], pos_vector.tolist(), ner_vector.tolist()]})
                print(word_dict)
            #range(i[0][0], i[0][1]) means the relation that we interest
            for j in data_dic['edges']:
                for key, value in j[2].items():
                    word_relations.append([word_dict[j[0][0]], word_dict[j[1][0]], key])
    with open('processed_' + input_file, 'w') as f:
        json.dump({'Relation':word_relations}, f)

    



if __name__ == '__main__':
    start = time.time()
    model = gensim.downloader.load('glove-twitter-100')
    print('model load time: ', time.time() - start)
    TrainData = 'train.json'
    TestData = 'test.json'



    ReadFile(TrainData, model)
