# IRIE final project: Relation Extraction 
Textual analogy parsing (TAP) is a task of identifying the analogical relations between texts. Given a sentence which contains a group of analogous facts, the results should represent the similiarity and difference of the given pairs of points in the texts.

For further information of textual analogy parsing:
 - [Textual Analogy Parsing: What's Shared and What's Compared among Analogous Facts](https://nlp.stanford.edu/pubs/lamm2018analogies.pdf) [[Github](https://github.com/mrlamm/textual-analogy-parsing)]

# Data
 * Tokens: Use the tokens order to match **index** in Nodes and Edges
 * Nodes: The type of the token. (18 kinds of labels)
 * Edges: The relation between tokens/Nodes.
# Task: Edge Prediction
 1. Given Tokens **without Nodes information**, predict the edge of given pair.
 2. Given Tokens and Nodes information, predict the edge of given pair.
# Evaluation
 *  We use F1 to evaluate the performance

For further information of the project: [final.pdf](https://github.com/lilinmail0523/IR-final-project2019-relation-extraction/blob/master/final.pdf)

# Word Embedding using pretrained model
 * FastText: [wiki-news-300d-1M.vec](https://fasttext.cc/docs/en/english-vectors.html)

# Data preprocessing:
 * Nodes: Node features were transformed to **one-hot encoding** and word embedding by FastText pretrained model.
 * POS: Part of Speech, fetched by nltk pos_tag (35 kinds of labels) function from **tokens**,  were transformed to **one-hot encoding** and **word embedding** by FastText pretrained model.
 * Word Vectors: Word vectors were fetch by FastText pretrained model from tokens.

# Model

 * 1-layer convolusiton

    ![1-layer convolution](https://github.com/lilinmail0523/IR-final-project2019-relation-extraction/blob/master/1-layer-convolution.png)
    
    Max: Max over channel, to determine the useful feature in each dimension
    
    Max over channel reference: [Relation Classiﬁcation via Convolutional Deep Neural Network](https://www.aclweb.org/anthology/C14-1220.pdf)
    | Hyper- parameters | Oprimizer  |Learning rate | Weight decay |Loss function| Epoch | Batch size |
    |---|---|---|---| ---| ---|---|
    | Value  | Adam  | 0.01 |0.00001 |Cross Entropy | 100 | 128 |


 * SVM (Multiclass by [OneVsRestClassifier](https://scikit-learn.org/stable/modules/generated/sklearn.multiclass.OneVsRestClassifier.html))
     
# Results
1. One hot encoding using SVM

    |   | Node  | POS  |
    |---|---|---|
    | Test f1 score  | 0.928  | 0.867  |
2. Word embedding using convolution and SVM

    | Test f1 score  | Node  |  POS | Word  | Node+POS+Word  |
    |---|---|---|---|---|
    | SVM  | 0.815  | 0.784  | 0.723  | 0.813  |
    | convolution  | 0.871  | 0.815  | 0.865  | 0.874  |

    Node+POS+Word vectors were combined by concatenation. 
