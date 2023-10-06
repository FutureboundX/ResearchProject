#Imports Word2Vec from gensim package, import word_tokenize from nltk
from gensim.models import Word2Vec
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

#Test Corpus
corpus = [
    "The Content Marketing Coordinator is responsible for designing various types of media content (print, digital, social etc.) to generate content and drive marketing campaigns",
    "This role will execute and organize daily administrative tasks and provide project support for the marketing team, including assisting with trade shows, managing creative requests, delivering exceptional customer support and collaborating with internal teams to achieve desired marketing deliverables.",
    "This position reports to the Marketing Communications Manager."
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
vector = model.wv["marketing"]
print("Embedding for 'marketing':", vector)

#Example: Finding similar words
similar_words = model.wv.most_similar("marketing", topn=3)
print("Words similar to 'marketing':", similar_words)
