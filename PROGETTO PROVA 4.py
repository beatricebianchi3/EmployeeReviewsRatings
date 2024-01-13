# PROGETTO PROVA 4 tutto il codice o quasi viene dal sito  https://towardsdatascience.com/multi-class-text-classification-model-comparison-and-selection-5eb066197568
# qui hai provato a usare un altro metodo per ripuire il testo che sembra molto giusto e poi hai appicato il 
# docuemnt embedding con doc2vec e regression model per la classifictaion 
import logging
import pandas as pd
import numpy as np
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
# path of the CSV file
csv_file_path = 'employee_reviews.csv'

# READ the CSV file into a DataFrame df
df = pd.read_csv(csv_file_path)
print(df['location'][0:5]) # this is how you have access to the column location and to the first 5 elements 
print(len(df)) # this is how you have access to the number of obs of the dataset 

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
print(len(employee_df)) # to show that I removed the missing values observations
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
    
employee_df['summary'] = employee_df['summary'].apply(clean_text)

print(employee_df['summary'][0:3])






# metodo che uso per risolvere il problema di tokenizer 
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
import pandas as pd
from nltk.tokenize import word_tokenize
import nltk

# Assicurati di aver scaricato i dati di tokenizzazione di nltk
nltk.download('punkt')



# Importa il modulo per la tokenizzazione
from nltk.tokenize import word_tokenize

# Aggiungi una colonna 'tokenized_text' al DataFrame
employee_df.loc[:, 'tokenized_text'] = employee_df['summary'].apply(lambda text: word_tokenize(str(text).lower()))

# Crea i documenti etichettati utilizzando 'overall-ratings' come tag
# Riempi i valori mancanti con un valore predefinito (ad esempio, 0)
#employee_df['overall-ratings'] = employee_df['overall-ratings'].fillna(0).astype(int)
#employee_df['overall-ratings'] = employee_df['overall-ratings'].astype(int)

labeled_data = [TaggedDocument(words=doc, tags=[str(tag)]) for doc, tag in zip(employee_df['tokenized_text'], range(len(employee_df['summary'])))]
print(len(labeled_data))
# Inizializza e addestra il modello Doc2Vec
vector_size = 100
window = 5
min_count = 1
workers = 4
epochs = 10

model = Doc2Vec(vector_size=vector_size, window=window, min_count=min_count, workers=workers, epochs=epochs)
model.build_vocab(labeled_data)
model.train(labeled_data, total_examples=model.corpus_count, epochs=model.epochs)

doc_vectors = [model.dv[idx] for idx in range(len(labeled_data))] 

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
print(f"Accuracy: {accuracy}") # 0.379
