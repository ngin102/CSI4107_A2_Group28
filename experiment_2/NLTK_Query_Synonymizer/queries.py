import os
import re
from typing import Dict
from stemmer import PorterStemmer


class Queries:
    def __init__(self):
        self.queries = {}
        self.stopwords = {}
        self.full_query = {}

        current_file_dir = os.path.dirname(os.path.abspath(__file__))
        file_path = os.path.join(current_file_dir, "topics1-50.txt")
        s = PorterStemmer()

        stopword_file = os.path.join(current_file_dir, "stopwords.txt")

        # Getting stopwords; ideally do this with a map
        with open(stopword_file) as f:
            for subword in f.read().split():
                stemmed_word = s.stem(subword, 0, len(subword) - 1)
                self.stopwords[stemmed_word] = 1

        if os.path.isfile(file_path):
            with open(file_path, "r") as file:
                content = file.read()

                # Split the file into individual queries based on the <top> tag
                text_queries = content.split("<top>")
                for i in range(1, len(text_queries)):
                    query = text_queries[i]

                    query = re.sub("<num>", "", query)
                    query = re.sub("<desc>(.*?)</top>", "", query)
                    query = re.sub("<title>", "", query)

                    query = query.lower()  # convert all characters to lowercase
                    query = re.sub(
                        "[^a-zA-Z0-9\\s]", "", query
                    )  # Remove all punctuation and other special characters
                    query = re.sub("\\s+", " ", query)  # Replace multiple spaces
                    query = re.sub("\\d", "", query)  # Remove numbers
                    print(query)
                    query = get_synonyms(query)
                    print(query)
                    query = query.strip()  # Remove trailing whitespaces

                    self.full_query[str(i)] = query

                    subwords = query.split(" ")

                    stemmed_query = ""
                    for j in range(len(subwords)):
                        if subwords[j] not in self.stopwords:
                            stemmed_word = s.stem(subwords[j], 0, len(subwords[j]) - 1)
                            stemmed_query += stemmed_word + " "

                    self.queries[str(i)] = stemmed_query.strip()
                    # self.full_query[str(i)] = stemmed_query.strip()

    def get_queries(self) -> Dict[str, str]:
        return self.queries

    def get_full_queries(self):
        return self.full_query

    def __str__(self) -> str:
        result = ""
        for key in self.queries.keys():
            result += f"{key}: {self.queries[key]}\n"
        return result

import importlib

# Check if NLTK is installed
try:
    importlib.import_module('nltk')
except ImportError:
    # Install NLTK
    import subprocess
    import sys
    print("Importing nltk!")
    subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'nltk'])

import nltk

try:
    nltk.download('wordnet')
    nltk.download('punkt')
except:
    print("Error downloading wordnet.")

from nltk.corpus import wordnet

def get_synonyms(sentence):
    # Tokenize the input sentence into a list of words
    tokens = nltk.word_tokenize(sentence)

    # Define an empty string to store synonyms of each word
    synonyms_string = ""

    # Loop through each word in the input sentence
    for token in tokens:
        # Define an empty set to store synonyms of the current word
        synonyms = set()
        for synset in wordnet.synsets(token):
            for lemma in synset.lemmas():
                # Only consider one-word synonyms
                if ('_' not in lemma.name()) and (' ' not in lemma.name()):
                    synonyms.add(lemma.name())
        synonyms_string += " ".join(synonyms)

    return synonyms_string
