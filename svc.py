import json
import os
import numpy as np
from sklearn.svm import SVC
from sklearn.multiclass import OneVsRestClassifier
from sklearn.model_selection import train_test_split

from sklearn.metrics import precision_recall_fscore_support
import time



def read_json(data_dir):
    """
    fetch data from preprocessed file
    sentence_id[0] : word1 information
    sentence_id[1] : word2 information
    sentence_id[2] : relation between two word
    word information:
    word[0] : word vector
    word[1] : node vector
    word[2] : location attribute
    word[3] : POS vector
    word[4] : NER vector
    """
    word= []
    node= []
    pos = []
    ner = []
    distance = []

    label = []

    with open(data_dir, 'r') as f:
        data_dic = json.loads(f.read())
        for sentence_id in data_dic['Relation']:
            word.append(sentence_id[0][0] + sentence_id[1][0])
            node.append(sentence_id[0][1] + sentence_id[1][1])
            pos.append(sentence_id[0][3] + sentence_id[1][3])
            ner.append(sentence_id[0][4] + sentence_id[1][4])
            distance.append([sentence_id[0][2] - sentence_id[1][2], sentence_id[1][2] - sentence_id[0][2]])

            if sentence_id[2] == 'fact':
                label.append(0)
            elif sentence_id[2] == 'equivalence':
                label.append(1)
            elif sentence_id[2] == 'analogy':
                label.append(2)

    return word, node, pos,  ner, distance, label



def SVM(train_x, train_y, val_x, val_y, test_x, test_y, type):
    """
    use One-Vs-The-Rest SVM for multiclass classification
    use precision_recall_fscore_support calculate f1 score
    (http://ogrisel.github.io/scikit-learn.org/sklearn-tutorial/modules/multiclass.html)
    These are only one feature cases
    """
    print('-'*20 + type + '-'*20)
    start = time.time()
    model = OneVsRestClassifier(SVC(C=1.0)).fit(train_x, train_y)

    train_pred = model.predict(train_x)
    acc, recall, f1, _ = precision_recall_fscore_support(train_y, train_pred, average= 'micro')
    print('Training f1 score: ', f1)

    val_pred = model.predict(val_x)
    acc, recall, f1, _ = precision_recall_fscore_support(val_y, val_pred, average= 'micro')
    print('Validation f1 score: ', f1)

    test_pred = model.predict(test_x)
    acc, recall, f1, _ = precision_recall_fscore_support(test_y, test_pred, average= 'micro')
    print('Test f1 score: ', f1)
    print('Total time of ' + type, time.time()- start)

def CombinationTest(train, validation, test, train_label, val_label, test_label,type):
    SVM(train,train_label,  validation, val_label, test, test_label, type)



if __name__ == '__main__':
    import warnings

    warnings.filterwarnings("ignore")

    #train data
    tr_word, tr_node, tr_pos,  tr_ner, tr_distance, tr_label = read_json('processed_train.json')
    #test data
    test_word, test_node, test_pos, test_ner, test_distance, test_label = read_json('processed_test.json')

    train_word, val_word, train_node, val_node, train_pos, val_pos, train_ner, val_ner, train_distance,val_distance, train_label, val_label  = train_test_split(
        tr_word, tr_node, tr_pos,  tr_ner, tr_distance, tr_label, test_size=0.20, random_state=42)



    #SVM(train_word,train_label,  val_word, val_label, test_word, test_label, 'word vector')
    #CombinationTest(np.concatenate((train_word, train_distance), axis=1), np.concatenate((val_word, val_distance), axis=1), np.concatenate((test_word, test_distance),axis=1), train_label, val_label, test_label, "word + distance")
   
    #SVM(train_node,train_label,  val_node, val_label, test_node, test_label, 'node vector')

    #SVM(train_pos,train_label,  val_pos, val_label, test_pos, test_label, 'pos vector')
    #CombinationTest(np.concatenate((train_pos, train_distance), axis=1), np.concatenate((val_pos, val_distance), axis=1), np.concatenate((test_pos, test_distance),axis=1), train_label, val_label, test_label, "pos + distance")

    '''
    test combination of word2vector, node, pos, ner, distance
    measure in 7 cases
    pos + ner (+ distance)
    word + pos (+ distance)
    word + pos + ner (+ distance)
    node + word + pos + ner + distance

    '''
    # pos + ner (+ distance) 
    #CombinationTest(np.concatenate((train_pos, train_ner), axis=1), np.concatenate((val_pos, val_ner), axis=1), np.concatenate((test_pos, test_ner),axis=1), train_label, val_label, test_label, "pos + ner")
    #CombinationTest(np.concatenate((train_pos, train_ner, train_distance), axis=1), np.concatenate((val_pos, val_ner, val_distance), axis=1), np.concatenate((test_pos, test_ner, test_distance),axis=1), train_label, val_label, test_label, "pos + ner + distance")
    # word + pos (+ distance) 
    #CombinationTest(np.concatenate((train_word, train_pos), axis=1), np.concatenate((val_word, val_pos), axis=1), np.concatenate((test_word, test_pos),axis=1), train_label, val_label, test_label, "word + pos")
    #CombinationTest(np.concatenate((train_word, train_pos, train_distance), axis=1), np.concatenate((val_word, val_pos, val_distance), axis=1), np.concatenate((test_word, test_pos, test_distance),axis=1), train_label, val_label, test_label, "word + pos + distance")
    # word + pos + ner (+ distance) 
    #CombinationTest(np.concatenate((train_word, train_pos, train_ner), axis=1), np.concatenate((val_word, val_pos, val_ner), axis=1), np.concatenate((test_word, test_pos, test_ner),axis=1), train_label, val_label, test_label, "word + pos + ner")
    #CombinationTest(np.concatenate((train_word, train_pos, train_ner, train_distance), axis=1), np.concatenate((val_word, val_pos, val_ner, val_distance), axis=1), np.concatenate((test_word, test_pos, test_ner, test_distance),axis=1), train_label, val_label, test_label, "word + pos + ner + distance")
    # node + word + pos + ner + distance 
    CombinationTest(np.concatenate((train_node, train_word, train_pos, train_ner, train_distance), axis=1), np.concatenate((val_node, val_word, val_pos, val_ner, val_distance), axis=1), np.concatenate((test_node, test_word, test_pos, test_ner, test_distance),axis=1), train_label, val_label, test_label, "node + word + pos + ner + distance ")

