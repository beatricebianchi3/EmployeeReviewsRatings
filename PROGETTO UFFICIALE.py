# PROGETTO UFFICIALE 

# DATASET 
# upload the dataset 
import numpy as np
import pandas as pd

# path of the CSV file
csv_file_path = 'employee_reviews.csv'

# READ the CSV file into a DataFrame df
df = pd.read_csv(csv_file_path)
print(df['location'][0:5]) # this is how you have access to the column location and to the first 5 elements 
print('length of the complete dataset', len(df)) # this is how you have access to the number of obs of the dataset 

# CLEAN the dataset 
# columns that I want to work with 
selected_columns = ['company', 'job-title', 'summary', 'pros', 'cons', 'overall-ratings']

# Create a new DataFrame with only the selected columns
employee_df = df[selected_columns]
print(employee_df[0:2]) # just to show how the dataset is 


# identify missing values
missing_values = employee_df.isnull().sum()
print('missing_values', missing_values) # I have 127 observations without summary so I remove them 

# remove rows with missing values
employee_df = employee_df.dropna()
print('length of the dataset withoyu missing values', len(employee_df)) # to show that I removed the missing values observations

# MORE CLEANING OF THE DATASET 
import logging
from numpy import random
import gensim
import nltk
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
from nltk.corpus import stopwords
import re
from bs4 import BeautifulSoup
import nltk

nltk.download('stopwords')

REPLACE_BY_SPACE_RE = re.compile('[/(){}\[\]\|@,;]')
BAD_SYMBOLS_RE = re.compile('[^0-9a-z #+_]')
STOPWORDS = set(stopwords.words('english'))

def clean_text(text):
    """
        text: a string
        
        return: modified initial string
    """
    text = BeautifulSoup(text, "lxml").text # HTML decoding
    text = text.lower() # lowercase text
    text = REPLACE_BY_SPACE_RE.sub(' ', text) # replace REPLACE_BY_SPACE_RE symbols by space in text
    text = BAD_SYMBOLS_RE.sub('', text) # delete symbols which are in BAD_SYMBOLS_RE from text
    text = ' '.join(word for word in text.split() if word not in STOPWORDS) # delete stopwors from text
    return text
    
# clean of the three columns with text ('summary','pros','cons')
employee_df['summary'] = employee_df['summary'].apply(clean_text)
employee_df['pros'] = employee_df['pros'].apply(clean_text)
employee_df['cons'] = employee_df['cons'].apply(clean_text)


# DIVIDE THE DATASET IN TRAINING AND TEST SETS 

# creating TRAINING and TEST sets 
from sklearn.model_selection import train_test_split
# group the DataFrame by the 'company' column
grouped_df = employee_df.groupby('company')

# empty DataFrames for training and test sets
train_set = pd.DataFrame(columns=selected_columns)
test_set = pd.DataFrame(columns=selected_columns)

# split each group into training and test sets with a mix of observations
for _, group_df in grouped_df:
    # test_size=0.2 => 20% in test set and 80% i training set 
    # random_state=42 => it is setting the seed ie data splitting will be the same every time you run the code
    group_train_set, group_test_set = train_test_split(group_df, test_size=0.2, random_state=42)

    
    # append the split sets to the overall training and test sets
    train_set = pd.concat([train_set, group_train_set], ignore_index=True)
    test_set = pd.concat([test_set, group_test_set], ignore_index=True)







# METHODS 


# 1) NAIVE CLASSIFIER 
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.pipeline import Pipeline

# Pipeline that combines text vectorization and the Multinomial Naive Bayes classifier:
# the first step ('vect') is to convert the text data into a numerical format using the CountVectorizer,
# and the second step ('clf') is the Multinomial Naive Bayes classifier.
text_clf = Pipeline([
    ('vect', CountVectorizer()),  # Step 1: Text vectorization
    ('clf', MultinomialNB())     # Step 2: Multinomial Naive Bayes classifier
])

print('lunghezza del test set', len(test_set))
#fit the model on the training dataset 
text_clf.fit(train_set['summary'], train_set['overall-ratings'])

# evaluate the model on the test set 
test_pred = text_clf.predict(test_set['summary'])


from collections import Counter
print("Prediction: ", Counter(test_pred))

print("Test set : ", Counter(test_set['overall-ratings']))

print(classification_report(test_set['overall-ratings'], test_pred,)) # ACCURACY 0.45

#precision    recall  f1-score   support

#         1.0       0.49      0.19      0.27       796
#         2.0       0.32      0.08      0.13      1055
#         3.0       0.35      0.24      0.29      2466
#         4.0       0.42      0.47      0.44      4522
#         5.0       0.51      0.66      0.58      4644

#    accuracy                           0.45     13483
#   macro avg       0.42      0.33      0.34     13483
#weighted avg       0.43      0.45      0.43     13483

'''
# 5) SUPPORT VECTOR MACHINE 

from sklearn.feature_extraction.text import TfidfVectorizer

vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(train_set['summary'])
'''
'''
from sklearn.decomposition import PCA

pca = PCA(n_components=2)
X_2d = pca.fit_transform(X.toarray())


import matplotlib.pyplot as plt
import seaborn as sns

plt.figure(figsize=(12, 8))
sns.scatterplot(x=X_2d[:, 0], y=X_2d[:, 1], hue=train_set['overall-ratings'], palette='viridis', legend='full', s=50)
plt.title('Summaries training data base on label "overall-ratings"')
plt.xlabel('First principal component')
plt.ylabel('Second principal component')
plt.show()

'''
'''
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.svm import SVC  # Import Support Vector Classification
from sklearn.metrics import classification_report

# Pipeline that combines text vectorization and the Support Vector Machine classifier:
# the first step ('vect') is to convert the text data into a numerical format using the CountVectorizer,
# and the second step ('clf') is the Support Vector Machine classifier.
text_clf2 = Pipeline([
    ('vect', CountVectorizer()),  # Step 1: Text vectorization
    ('clf', SVC(kernel='linear', C=1.0))                 # Step 2: Support Vector Machine classifier
])

# Fit the model on the training dataset
text_clf2.fit(train_set['summary'], train_set['overall-ratings'])

# Evaluate the model on the test set
test_pred2 = text_clf2.predict(test_set['summary'])

from collections import Counter
print("Prediction: ", Counter(test_pred2))
print("Test set : ", Counter(test_set['overall-ratings']))

print(classification_report(test_set['overall-ratings'], test_pred2)) # ACCURACY 0.45

#precision    recall  f1-score   support

#         1.0       0.49      0.29      0.36       796
#         2.0       0.27      0.08      0.13      1055
#         3.0       0.37      0.23      0.28      2466
#         4.0       0.42      0.46      0.44      4522
#         5.0       0.50      0.67      0.57      4644

#    accuracy                           0.45     13483
#   macro avg       0.41      0.35      0.36     13483
#weighted avg       0.43      0.45      0.43     13483
'''

from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from nltk.tokenize import word_tokenize
import nltk

# Assicurati di aver scaricato i dati di tokenizzazione di nltk
nltk.download('punkt')

# Importa il modulo per la tokenizzazione
from nltk.tokenize import word_tokenize

# Aggiungi una colonna 'tokenized_text' al DataFrame
employee_df.loc[:, 'tokenized_text'] = employee_df['summary'].apply(lambda text: word_tokenize(str(text).lower()))

# Crea i documenti etichettati utilizzando 1,2,3,4.... come tag
labeled_data = [TaggedDocument(words=doc, tags=[str(tag)]) for doc, tag in zip(employee_df['tokenized_text'], range(len(employee_df['summary'])))]

# Inizializza e addestra il modello Doc2Vec
vector_size = 100
window = 5
min_count = 1
workers = 4
epochs = 10

model = Doc2Vec(vector_size=vector_size, window=window, min_count=min_count, workers=workers, epochs=epochs)
model.build_vocab(labeled_data)
model.train(labeled_data, total_examples=model.corpus_count, epochs=model.epochs)

# list of all vector connected to each summary 
doc_vectors = [model.dv[idx] for idx in range(len(labeled_data))] 


# applying logisticregression classifier 
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(doc_vectors, employee_df['overall-ratings'], test_size=0.2, random_state=42)

# Initialize and train a logistic regression classifier
classifier = LogisticRegression(max_iter=1000)
classifier.fit(X_train, y_train)

# Make predictions on the test set
y_pred = classifier.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy}") # ACCURACY  0.39


# ora devi rappresentare i risultati che hai trovato ovvero devi mostrare il test set con i risultati trovati dai modelli
# quindi divisi per company e divisi per valore di overall_ratings

import matplotlib.pyplot as plt
import seaborn as sns

# Assuming 'company' is the name of the column in your DataFrame
# Replace it with the actual name if different
company_column = 'company'

# Creare un DataFrame con le previsioni e le aziende corrispondenti nel test set
predictions_df = pd.DataFrame({'Predicted_Rating': test_pred, 'Company': test_set[company_column]})

# Creare un bar plot delle predizioni divise per azienda
plt.figure(figsize=(14, 8))
sns.countplot(x='Company', hue='Predicted_Rating', data=predictions_df, palette='viridis', dodge=True)
plt.title('Predicted Ratings Distribution by Company')
plt.xlabel('Company')
plt.ylabel('Count')
plt.legend(title='Predicted Rating', loc='upper right')
plt.show()


# ora calcolerei tutte le perentuali di 1,2,3,4,5 in ogni azienda per capire in generale se l'azienda è efficiente o no 
# Calcola il numero totale di previsioni per ogni azienda e livello
company_level_counts = predictions_df.groupby(['Company', 'Predicted_Rating']).size().unstack(fill_value=0)

# Calcola le percentuali
company_level_percentages = company_level_counts.div(company_level_counts.sum(axis=1), axis=0) * 100

# Stampa le percentuali
print(company_level_percentages)


# Unisci le percentuali di 1 e 2 in una colonna e di 4 e 5 in un'altra colonna
company_level_percentages['Low Ratings'] = company_level_percentages[[1, 2]].sum(axis=1)
company_level_percentages['High Ratings'] = company_level_percentages[[4, 5]].sum(axis=1)

# Stampa le percentuali aggregate
print(company_level_percentages[['Low Ratings', 'High Ratings']])

'''
Predicted_Rating       1.0       2.0        3.0        4.0        5.0
Company
amazon            3.486169  2.482001  17.127700  37.912088  38.992042
apple             1.431335  1.934236   8.510638  34.197292  53.926499
facebook          1.577287  0.630915   4.100946  21.135647  72.555205
google            0.768738  1.217168   6.854580  30.172966  60.986547
microsoft         1.480034  1.843061  12.901424  45.573862  38.201620
netflix           8.074534  6.832298  12.422360  26.708075  45.962733
Predicted_Rating  Low Ratings  High Ratings
Company
amazon               5.968170     76.904130
apple                3.365571     88.123791
facebook             2.208202     93.690852
google               1.985906     91.159513
microsoft            3.323094     83.775482
netflix             14.906832     72.670807
'''