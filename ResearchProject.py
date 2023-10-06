#Imports Word2Vec from gensim package, import word_tokenize from nltk
from gensim.models import Word2Vec
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

#Test Corpus
corpus = [
    "I love natural language processing.",
    "Word embeddings are fascinating.",
    "Machine learning is important for AI."
]

#Tokenize the sentences and remove stopwords
stop_words = set(stopwords.words("english"))
sentences = [word_tokenize(sentence.lower()) for sentence in corpus]
sentences = [[word for word in sentence if word.isalnum() and word not in stop_words] for sentence in sentences]

#Create and train the Word2Vec model
model = Word2Vec(sentences, vector_size=100, window=5, min_count=1, sg=0)

#Save the trained model
model.save("word2vec_model")

#Example: Getting word embeddings
vector = model.wv["love"]
print("Embedding for 'love':", vector)

#Example: Finding similar words
similar_words = model.wv.most_similar("love", topn=3)
print("Words similar to 'love':", similar_words)
