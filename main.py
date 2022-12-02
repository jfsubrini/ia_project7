# -*- coding: utf-8 -*-
"""
Created by Jean-François Subrini on the 29th of November 2022.
Creation of a simple sentiment analysis REST API 
using the FastAPI framework and a LSTM model (created in the Notebook 2).
This REST API has been deployed on Heroku.
"""
### IMPORTS ###
# Importation of Python modules and methods.
import re
import string

# Importation of libraries.
import contractions
import numpy as np
import pickle5 as pickle
import uvicorn
from fastapi import FastAPI
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import sequence

# Importation of nltk functions or classes.
import nltk
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import TweetTokenizer
###---###

# Loading the selected LSTM model.
lstm_model = load_model('model_lstm2')

# Creating the app object.
app = FastAPI()


### UTIL FUNCTIONS ###
def preprocess_text(text):
    """Function to preprocess text data. Pipeline of actions to reduce the dimension
    (number of tokens), 'clean' and send a list of final tokens.
    """
    # Dropping URLs.
    new_corpus = re.sub(r"http\S+", "", text)
    
    # Dropping the articles, conjunctions, proper nouns, interjections... Part-Of-Speech.
    # CD: Cardinal number | DT: Determiner | IN: Preposition or subordinating conjunction | 
    # MD: Modal | NNP: Proper noun, singular | NNPS: Proper noun, plural | SYM: Symbol | 
    # UH: Interjection | WDT: Wh-determiner.
    # POS list of tags to avoid.
    tags_to_avoid = ['CD', 'DT', 'IN', 'MD', 'NNP', 'NNPS', 'SYM', 'UH', 'WDT']
    tags = nltk.pos_tag(new_corpus.split())
    tagged_tokens = [w for w in tags if w[1] not in tags_to_avoid]
    
    # Untagged tokens transformed to string.
    untagged_corpus = " ".join([w[0] for w in tagged_tokens])
    
    # Lowercasing words.
    lower_corpus = untagged_corpus.lower()

    # Specific tweet tokenization.
    tknzr = TweetTokenizer(preserve_case=False, strip_handles=True)
    raw_tokens_list = tknzr.tokenize(lower_corpus)

    # Excluding all tokens with a numeric character.
    alpha_tokens = [w for w in raw_tokens_list if not w.isdigit()]

    # Dropping punctuation.
    tokens_without_punct = [w for w in alpha_tokens if w not in string.punctuation]

    # Expanding contractions, the shortened words.
    tokens_exp = [contractions.fix(word) for word in tokens_without_punct]

    # Mapping for some undetected contractions.
    mapping = {"dont": "do not", "cant": "cannot", "nt": "not", 
               "gd": "good", "gud": "good", "shoulda": "should not", "ca": "can", 
               "coulda": "could not", "wana": "want to", "da": "that",
               "wanna": "want to", "gonna": "going to", "sorta": "sort of"}
    tokens_expand = [mapping.get(item, item) for item in tokens_exp]
    
    # Normalization with lemmatization. Reduce words to their root form.
    lem = WordNetLemmatizer()
    final_tokens = [lem.lemmatize(i, pos='v') 
                   for i in tokens_expand]  # Part-Of-Speech tag with verb.
    
    return final_tokens

def generate_vector(cleaned_doc):
    """Generate a vector out of the cleaned text, with zeros and integers."""
    # Loading the trained tokenizer of the selected LSTM model.
    with open('tokenizer.pickle', 'rb') as file:
        tokenizer = pickle.load(file)
    encoded_doc = tokenizer.texts_to_sequences([cleaned_doc])  # list expected
    
    # Padding the tweet to a max length of 227 words, with 'pre' zeros.
    doc_vector = sequence.pad_sequences(encoded_doc, maxlen=227, padding='pre')[0]
    
    return doc_vector
###---###

# Index route, opens automatically on http://127.0.0.1:8000.
@app.get('/')
def index():
    """Welcome message"""
    return {'message': 'This is a sentiment analysis app.'}

# Route with a string parameter (text), returns the sentiment prediction of the text.
# Located at: http://127.0.0.1:8000/sentiment_analysis/?text=
# Also access to the FastAPI swagger to type directly the text to analyse.
# Located at: http://127.0.0.1:8000/docs
@app.get('/sentiment_analysis/')
def predict_sentiment(text: str):
    """Get and process result to predict a binary sentiment analysis on a text."""
    # Getting the cleaned text with tokens.
    final_tokens = preprocess_text(text)
    cleaned_doc = " ".join(final_tokens)
    
    # Vectorizing the cleaned_doc.
    doc_vector = generate_vector(cleaned_doc)

    # Predicting the result.
    result = lstm_model(np.array(doc_vector, ndmin=2))  # quicker than with predict, because no batch.
    if result.numpy()[0][0] < 0.5:
        sent = 'Negative sentiment'
    else:
        sent = 'Positive sentiment'

    return {'sentiment': sent}


# Running the API with uvicorn.
# Will run on http://127.0.0.1:8000
if __name__ == '__main__':
    uvicorn.run(app, host='127.0.0.1', port=8000)
    
