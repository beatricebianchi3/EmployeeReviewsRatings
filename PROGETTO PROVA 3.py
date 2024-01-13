# PROVA 3 tutto il codice viene da https://spotintelligence.com/2023/09/06/doc2vec/
# qui hai sviluppato come utilizzare il metodo doc2vec per tokenizzare le parole e creare dei vettori per le paorle 
# poi per la classification hai usato regression 

from gensim.models.doc2vec import Doc2Vec, TaggedDocument
# Sample documents (replace with your own data)
# upload the dataset 
import numpy as np
import pandas as pd

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
print(f"Accuracy: {accuracy}") # 0.39



# per vedere se riesci a applicare il supprt vector machine su questo in modo bello 

# PCA
from sklearn.decomposition import PCA
pca = PCA(n_components=2)
X_2d = pca.fit_transform(X_train)

# Visualizzazione
import matplotlib.pyplot as plt
import seaborn as sns

plt.figure(figsize=(12, 8))
sns.scatterplot(x=X_2d[:, 0], y=X_2d[:, 1], hue=y_train, palette='viridis', legend='full', s=50)
plt.title('Summaries training data based on label "overall-ratings"')
plt.xlabel('First principal component')
plt.ylabel('Second principal component')
plt.show()













'''
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from nltk.tokenize import word_tokenize
import pandas as pd
import nltk

# Aggiungi questa linea per caricare il tokenizer di NLTK
nltk.download('punkt')

# Definisci una funzione per tokenizzare il testo
def tokenize_text(text):
    return word_tokenize(text.lower())

employee_df['summary'] = employee_df['summary'].fillna('')


# Aggiungi una colonna 'tag' con gli indici delle righe come tag
employee_df['tag'] = employee_df.index.astype(str)

# Crea una lista di documenti con i testi delle summary e i tag corrispondenti
documents = [TaggedDocument(words=tokenize_text(row['summary']), tags=[row['tag']]) for index, row in employee_df.iterrows()]

# Inizializza il modello Doc2Vec
model = Doc2Vec(vector_size=100, window=2, min_count=1, workers=4, epochs=20)

# Costruisci il vocabolario
model.build_vocab(documents)

# Addestra il modello
model.train(documents, total_examples=model.corpus_count, epochs=model.epochs)

doc_vectors = [model.dv[idx] for idx in range(len(labeled_data))] 


tagged_data = []

# Iterare attraverso i documenti nel DataFrame e creare i TaggedDocument
for i, doc in enumerate(employee_df['summary']):
    tagged_data.append(TaggedDocument(words=doc.split(), tags=['doc'+str(i)]))

# Initialize the Doc2Vec model
model = Doc2Vec(vector_size=100, # Dimensionality of the document vectors 
                     window=5, # Maximum distance between the current and predicted word within a sentence 
                     min_count=1, # Ignores all words with total frequency lower than this 
                     workers=4, # Number of CPU cores to use for training 
                     epochs=10) # Number of training epochs
# Build the vocabulary
model.build_vocab(tagged_data)
# Train the model
model.train(tagged_data, total_examples=model.corpus_count, epochs=model.epochs)

doc_vectors = [model.dv[idx] for idx in range(len(tagged_data))] 
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(doc_vectors, employee_df['overall-ratings'], test_size=0.2, random_state=42)

# Initialize and train a logistic regression classifier
classifier = LogisticRegression()
classifier.fit(X_train, y_train)

# Make predictions on the test set
y_pred = classifier.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy}")

#Evaluate your text classification model using appropriate evaluation metrics (e.g., accuracy, precision, recall, F1-score). Depending on the results, you may need to fine-tune the Doc2Vec model hyperparameters or the classification algorithm for better performance.

#7. Predict New Documents:

#Once your model is trained and evaluated, you can classify new, unlabeled documents by transforming them into Doc2Vec embeddings and then using the trained classifier for predictions.

#By combining Doc2Vec embeddings with a text classification model, you can effectively classify documents based on their content, making it a valuable approach for tasks like sentiment analysis, topic classification, and document categorization.
'''