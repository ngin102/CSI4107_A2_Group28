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
