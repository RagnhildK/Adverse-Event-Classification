# Adverse-Event-Classification

## EDA 

## Classification of Adverse Events
The files in this projcet presents the code for the best performing Multinominal Naive Bayes and Support Vector Machine models, as presented in the thesis. 
The final models are found in files ```final_mnb.ipynb``` and ```final_svm.ipynb```. The dataset used have been exluded due to privacy and ethical reasons. The results are still shown in the files.  


### Other combinations tested
As presented in the thesis several combinations of preprocessing have been tested to acheive the best results. The code for other combinations than the ones evaluated in the thesis are not included, but please contact us if interested. 

The necessary imports used to test all combinations are as follows: 

````
import pandas as pd
import nltk as nltk
from nltk.corpus import stopwords
from nltk.stem.snowball import NorwegianStemmer
from nltk.tokenize import word_tokenize
from nltk.tokenize.treebank import TreebankWordDetokenizer
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.feature_selection import SelectPercentile, chi2, f_classif, mutual_info_classif
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB, ComplementNB, BernoulliNB
from sklearn import svm
import re
from sklearn import preprocessing
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, ConfusionMatrixDisplay, recall_score, precision_score, classification_report
```
