from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import re
from gensim.models import Word2Vec
import random

# Preparation of a text corpus
filename = 'path/example.txt'
file = open(filename, 'rt')
text = file.read()
file.close()

# Removal of punctuation marks and numbers
text = re.sub(r'[^a-zA-Z]', ' ', text)

# Converting the text to lowercase
text = text.lower()

# Tokenization of text into words
tokens = word_tokenize(text)

# Removal of stop words
stop_words = set(stopwords.words('english'))
filtered_tokens = [token for token in tokens if token not in stop_words]

# Lemmatization
lemmatizer = WordNetLemmatizer()
lemmatized_text = [lemmatizer.lemmatize(token) for token in filtered_tokens]

# Creating a Word Embeddings model
model = Word2Vec([lemmatized_text], min_count=1)

# Example: model chooses a random word from a text and finds the top-10 similar to it
word = random.choice(lemmatized_text)
similar_words = model.wv.most_similar(word, topn=10)

print(f"The most similar words to '{word}':")
for w, s in similar_words:
    print(f"{w}: {s}")

# Example: model finds words that are similar to a combination of positive words and dissimilar to negative words
words = random.sample(model.wv.index_to_key, 3)
word_1, word_2, word_3 = words[0], words[1], words[2]

similar_words = model.wv.most_similar_cosmul(positive=[word_1, word_2], negative=[word_3], topn=5)

print(f"\nPositive words: {word_1}, {word_2}. Negative word: {word_3}.")
print("Similar words:")
for word, similarity in similar_words:
    print(f"{word}: {similarity}")

# Example: model finds an odd word in a row of words randomly chosen from text
words_for_odd = random.sample(model.wv.index_to_key, 4)
odd_word = model.wv.doesnt_match(words_for_odd)
print(f"\nOdd word in a set {words_for_odd}: {odd_word}")