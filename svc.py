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
    x[0] : word1 information
    x[1] : word2 information
    x[2] : relation between two word
    word information:
    word[0] : word vector
    word[1] : node vector
    word[2] : location attribute
    word[3] : POS vector
    """
    word_vector = []
    pos_vector = []
    node_vector = []
    word2_pos = []
    label = []

    with open(data_dir, 'r') as f:
        data_dic = json.loads(f.read())
        for sentence_id in data_dic['Relation']:
            word_vector.append(sentence_id[0][0] + sentence_id[1][0])
            node_vector.append(sentence_id[0][1] + sentence_id[1][1])
            pos_vector.append(sentence_id[0][3] + sentence_id[1][3])

            if sentence_id[2] == 'fact':
                label.append(0)
            elif sentence_id[2] == 'equivalence':
                label.append(1)
            elif sentence_id[2] == 'analogy':
                label.append(2)

    return word_vector, node_vector, pos_vector,  label



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



if __name__ == '__main__':
    import warnings

    warnings.filterwarnings("ignore")

    #train data
    tr_word, tr_node, tr_pos,  tr_label = read_json('Processtrain.json')
    #test data
    test_word, test_node, test_pos, test_label = read_json('Processtest.json')

    train_word, val_word, train_node, val_node, train_pos, val_pos, train_label, val_label = train_test_split(tr_word, tr_node, 
                                                        tr_pos, tr_label, test_size=0.20, random_state=42)


    SVM(train_word,train_label,  val_word, val_label, test_word, test_label, 'word vector')
    
    SVM(train_node,train_label,  val_node, val_label, test_node, test_label, 'node vector')

    SVM(train_pos,train_label,  val_pos, val_label, test_pos, test_label, 'pos vector')

    train_combine = np.concatenate((train_word, train_node, train_pos), axis=1)
    val_combine = np.concatenate((val_word, val_node, val_pos), axis=1)
    test_combine = np.concatenate((test_word, test_node, test_pos), axis=1)
    SVM(train_combine,train_label,  val_combine, val_label, test_combine, test_label, 'Combine')

