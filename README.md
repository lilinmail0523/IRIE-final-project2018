# IRIE final project: Relation Extraction 
The aim of relation extraction is to identify the semantic relations in the articles like “at”, “role”, “social” and so on, and it could process the unstructured texts into organized information. In this project, the task of Textual Analogy Parsing was to find the analogous facts among the entities from the Wall Street Journals. The supervised method was used by extracting features including word embedding, part-of-speech (POS), named entity recognition (NER), Position feature (PF), and then applying SVM to acquire the relations. The evaluation was done by F1-score.

# Data
 * Tokens: Using the tokens order to match **index** in Nodes and Edges.
 * Nodes: The type of the token. (18 kinds of labels)
 * Edges: The relation between tokens/Nodes including fact, equivalence, and analogy.
# Task: Edge Prediction
 1. Given Tokens and Nodes information, predict the edge of given pair.
 2. Given Tokens **without Nodes information**, predict the edge of given pair.


# Method:
With supervised learning, first, the data were processed into POS, NER, PF, and WE features. The POS and NER features were presented by one-hot encoding, and WE was fetched from [Stanford GloVe twitter 100d model](https://nlp.stanford.edu/projects/glove/). Second, SVM was used to identify the relationships between tokens. 
  | Features  | Description |
  |---|---|
  | Node | The Quantitative Semantic Role Label (QSRL) given from task|
  | WE (Word embedding) | Word embedding extracted from GloVe twitter 100d|
  | Part-of-speech (POS) | Part of speech of each word|
  | Name entity recognition (NER) | Named entities classified into pre-define categories|
  | Position feature (PF)| Distance between two words|
  
# Results
1. Task one: with node features 

    |   | Node  | Node, WE, POS, PF  | WE, POS, NER, PF|
    |---|---|---|---|
    | Test f1 score  | 0.969  | 0.927  | 0.904 |
 
In the fist task, the results showed that node features had higher performance than the others, and the scores decreased when node features were concatenated with others. Node features were expressed by Quantitative Semantic Role Labels (QSRL) which provided tokens with roles, including quantitative language and emphasizes context. Compared to POS and WE, the QSRL offered more specific information that can divide data into several categories. For example, The value, time, reference_time labels in QSRL could differentiate the numbers which were all abbreviated as “CD” in POS method.  
    
2. Task two: without node features

    | Test f1 score  | WE  |  POS | POS, NER  | WE, POS, NER  |
    |---|---|---|---|---|
    | Without PF | 0.886  | 0.792  | 0.790  | 0.884  |
    | With PF | 0.898  | 0.856  | 0.857  | 0.901  |

The comparison of features in the second task indicated that merged features achieved the highest performance which took advantage of word meanings and part-of-speech. The fact edge was built from values and the relation of analogy and equivalence needed fact edge information to establish. The WE and POS features led to better performance because word meanings and part-of-speech could tell difference between value and non-value entities, while NER had nothing to do with numbers in the text. The features concatenated with PF improved performance because the equivalence and analogy edge usually crossed the sentence which had the larger distance, while the fact edge was bounded in one sentence with small distance.

# Reference

[Textual Analogy Parsing: What's Shared and What's Compared among Analogous Facts, Lamm *et al.*, 2018.](https://nlp.stanford.edu/pubs/lamm2018analogies.pdf) [[Github]](https://github.com/mrlamm/textual-analogy-parsing)
