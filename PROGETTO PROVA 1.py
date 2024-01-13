# PROGETTO PROVA 1 qui hai provato con tokenizer delle lezioni e usi sia naive bayes sia support vector machine 

# DATASET 
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
print(employee_df[0:2]) # just to show how the dataset is 


# identify missing values
missing_values = employee_df.isnull().sum()
print('missing_values', missing_values) # I have 127 observations without summary so I remove them 

# remove rows with missing values
employee_df = employee_df.dropna()
print(len(employee_df)) # to show that I removed the missing values observations



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


print(len(train_set))
print(train_set.iloc[1])
print(train_set.iloc[33200])
print(len(test_set))

# questo serve sono per controllare che venga tutto bene ie che io abbia divisi i dati in training e test set in modo che le percentuali siano calolate sui singoli gruppi e non su tutto il dataset insieme 
count_google = employee_df['company'].value_counts().get('google', 0)
count_google_train=  train_set['company'].value_counts().get('google', 0)
count_google_test=  test_set['company'].value_counts().get('google', 0)

print(f"Numero di elementi con il label 'google': {count_google}")
print(f"Numero di elementi con il label 'google'nel train set: {count_google_train}")
print(f"Numero di elementi con il label 'google'nel test set : {count_google_test}")







#funziona ma non farlo stampare tutte le volte quando ti servirà per il progetto togli il commento  
# VISUALIZE the dataset in two barplot for train and test set 
import matplotlib.pyplot as plt

train_set_count_ratings = train_set['overall-ratings'].value_counts()
print(train_set_count_ratings)
train_set_count_ratings.plot(x='count', y='dtype', kind='bar')
plt.title('Bar Plot TRAINING SET')
plt.xlabel('overall-ratings')
plt.ylabel('distribution')
plt.show()

test_set_count_ratings = test_set['overall-ratings'].value_counts()
print(test_set_count_ratings)
test_set_count_ratings.plot(x='count', y='dtype', kind='bar')
plt.title('Bar Plot TEST SET')
plt.xlabel('overall-ratings')
plt.ylabel('distribution')
plt.show()




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

print(classification_report(test_set['overall-ratings'], test_pred,))



'''
#   QUESTO DA GLI STESSI IDENTIFIC RISULTATI DEL NAIVE BAYES QUINDI NON ENSO SIA MOLTO UTILE 
# 2) DECISION TREE J48

from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report
from collections import Counter

# Create a Pipeline that combines text vectorization and the J48 (Decision Tree) classifier:
# The first step ('vect') is to convert the text data into a numerical format using CountVectorizer,
# and the second step ('clf') is the Decision Tree classifier (J48).
text_clf2 = Pipeline([
    ('vect', CountVectorizer()),       # Step 1: Text vectorization
    ('clf', DecisionTreeClassifier())  # Step 2: J48 (Decision Tree) classifier
])

# Fit the model on the training dataset
text_clf2.fit(train_set['summary'], train_set['overall-ratings'])

# Evaluate the model on the test set
test_pred2 = text_clf2.predict(test_set['summary'])

# Display prediction and actual class distribution
print("Prediction: ", Counter(test_pred2))
print("Test set : ", Counter(test_set['overall-ratings']))

# Display classification report
print(classification_report(test_set['overall-ratings'], test_pred2))

'''



''' 
# IL PROBLEMA DI QUESTO è CHE PREVEDE TUTTI IN 4 
# 3) SENTICFR 
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn_crfsuite import CRF
from sklearn_crfsuite.metrics import flat_classification_report
from collections import Counter
import pandas as pd


# Trasforma le valutazioni numeriche in etichette categoriche
train_set['overall-ratings'] = pd.cut(train_set['overall-ratings'], bins=[0, 1,2,3, 4, 5], labels=['1', '2','3','4', '5'])
print(train_set['overall-ratings'][0:5])
# Crea una pipeline con CountVectorizer e CRF
text_crf_clf = Pipeline([
    ('vect',  CountVectorizer()),            # Passo 1: Vettorizzazione del testo
    ('crf', CRF(algorithm='lbfgs'))         # Passo 2: Modello CRF 
])


# Addestra il modello sul set di addestramento
text_crf_clf.fit(train_set['summary'], train_set['overall-ratings'])


# evaluate the model on the test set 
test_pred3 = text_crf_clf.predict(test_set['summary'])

from itertools import chain

print(test_pred3[0:5])
flat_test_pred3 = list(chain.from_iterable(test_pred3))

from collections import Counter
print("Prediction: ", Counter(flat_test_pred3))

print("Test set : ", Counter(test_set['overall-ratings']))

print(classification_report(test_set['overall-ratings'], test_pred3))

'''


''' ANCHE QUESTO PREDICE SEMPRE LO STESSO VALORE 
# 4) ONER 

from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import classification_report

from sklearn.base import BaseEstimator, ClassifierMixin
from scipy.sparse import issparse
import numpy as np

class OneR(BaseEstimator, ClassifierMixin):
    def __init__(self):
        self.feature_rules = {}

    def fit(self, X, y):
        # Ensure that the target variable is of integer type
        y = y.astype(int)
        self.classes_ = np.unique(y)

        if issparse(X):
            X = X.toarray()

        for feature_idx in range(X.shape[1]):
            feature = X[:, feature_idx]
            rule = self._find_best_rule(feature, y)
            self.feature_rules[feature_idx] = rule

    def _find_best_rule(self, feature, target):
        # Convert feature values to integers
        feature = feature.astype(int)

        # Implement logic to find the best rule for the feature
        # This is a simplified example; you may need to customize it based on your data
        rule_values = np.unique(feature)
        best_rule = None
        best_accuracy = 0

        for value in rule_values:
            mask = (feature == value)
            accuracy = np.sum(target[mask] == np.argmax(np.bincount(target[mask]))) / np.sum(mask)

            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_rule = value

        return best_rule

    def predict(self, X):
        if issparse(X):
            X = X.toarray()

        predictions = np.zeros(X.shape[0], dtype=int)

        for feature_idx, rule in self.feature_rules.items():
            feature = X[:, feature_idx]
            # Convert predicted values to integers
            predictions[feature == rule] = np.argmax(np.bincount(self.classes_)).astype(int)

        return predictions


# Pipeline that combines text vectorization and the OneR classifier:
# the first step ('vect') is to convert the text data into a numerical format using the CountVectorizer,
# and the second step ('clf') is the OneR classifier.
text_clf4 = Pipeline([
    ('vect', CountVectorizer()),  # Step 1: Text vectorization
    ('clf', OneR()),               # Step 2: OneR classifier
])

# Fit the model on the training dataset
text_clf4.fit(train_set['summary'], train_set['overall-ratings'])

# Evaluate the model on the test set
test_pred4 = text_clf4.predict(test_set['summary'])


from collections import Counter
print("Prediction: ", Counter(test_pred4))
print("Test set : ", Counter(test_set['overall-ratings']))

print(classification_report(test_set['overall-ratings'], test_pred4))
'''

# 6)  NEAREST CENTROID 
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.neighbors import NearestCentroid
from sklearn.metrics import classification_report

# Pipeline that combines text vectorization and the Nearest Centroid classifier:
# the first step ('vect') is to convert the text data into a numerical format using the CountVectorizer,
# and the second step ('clf') is the Nearest Centroid classifier.
text_clf3 = Pipeline([
    ('vect', CountVectorizer()),        # Step 1: Text vectorization
    ('clf', NearestCentroid())          # Step 2: Nearest Centroid classifier
])

# Fit the model on the training dataset
text_clf3.fit(train_set['summary'], train_set['overall-ratings'])

# Evaluate the model on the test set
test_pred3 = text_clf3.predict(test_set['summary'])

from collections import Counter
print("Prediction: ", Counter(test_pred3))
print("Test set : ", Counter(test_set['overall-ratings']))

print(classification_report(test_set['overall-ratings'], test_pred3))



# 5) SUPPORT VECTOR MACHINE 

from sklearn.feature_extraction.text import TfidfVectorizer

vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(train_set['summary'])

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

print(classification_report(test_set['overall-ratings'], test_pred2))
'''


