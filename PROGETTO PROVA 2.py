#prova 2 
# tutto il codice viene dal sito https://carpentries-incubator.github.io/python-text-analysis/07-wordEmbed_intro/index.html


import gensim.downloader as api

# takes 3-10 minutes to load
#pre-trained model we loaded earlier was trained on a Google News dataset (about 100 billion words). We loaded this model as the variable wv earlier.
wv = api.load('word2vec-google-news-300') # takes 3-10 minutes to load 
print(type(wv))
# output : <class 'gensim.models.keyedvectors.KeyedVectors'>
# Gensim stores “KeyedVectors” representing the Word2Vec model. They’re called keyed vectors because you can use words as keys to extract the corresponding vectors. Let’s take a look at the vector representaton of whale.

print(wv['whale'])
print(wv['whale'].shape) 
#In this model, each word has a 300-dimensional representation. You can think of these 300 dimensions as 300 different features that encode a word’s meaning. Unlike LSA, which produces (somewhat) interpretable features (i.e., topics) relevant to a text, the features produced by Word2Vec will be treated as a black box. That is, we won’t actually know what each dimension of the vector represents. However, if the vectors have certain desirable properties (e.g., similar words produce similar vectors), they can still be very useful. Let’s check this with the help of the cosine similarity measure.

#cosine Similarity (Review): Recall from earlier in the workshop that cosine similarity helps evaluate vector similarity in terms of the angle that separates the two vectors, irrespective of vector magnitude. It can take a value ranging from -1 to 1, with…

#1 indicating that the two vectors share the same angle
#0 indicating that the two vectors are perpendicular or 90 degrees to one another
#-1 indicating that the two vectors are 180 degrees apart.
#Words that occur in similar contexts should have similar vectors/embeddings. How similar are the word vectors representing whale and dolphin?
print(wv.similarity('whale','dolphin'))

#Our similarity scale seems to be on the right track. We can also use the similarity function to quickly extract the top N most similar words to whale.
print(wv.most_similar(positive=['whale'], topn=10))

#Use Gensim’s most_similar function to find the top 10 most similar words to each of the following words (separately): “bark”, “pitcher”, “park”. Note that all of these words have multiple meanings depending on their context. Does Word2Vec capture the meaning of these words well? Why or why not?

#Solution
#Based on these three lists, it looks like Word2Vec is biased towards representing the predominant meaning or sense of a word. In fact, the Word2Vec does not explicitly differentiate between multiple meanings of a word during training. Instead, it treats each occurrence of a word in the training corpus as a distinct symbol, regardless of its meaning. As a result, resulting embeddings may be biased towards the most frequent meaning or sense of a word. This is because the more frequent a word sense appears in the training data, the more opportunities the algorithm has to learn its representation.

#Note that while this can be a limitation of Word2Vec, there are some techniques that can be applied to incorporate word sense disambiguation. One common approach is to train multiple embeddings for a word, where each embedding corresponds to a specific word sense. This can be done by pre-processing the training corpus to annotate word senses, and then training Word2Vec embeddings separately for each sense. This approach allows Word2Vec to capture different word senses as separate vectors, effectively representing the polysemy of the word.



