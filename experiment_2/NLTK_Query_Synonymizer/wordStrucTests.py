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
except:
    print("Error downloading wordnet.")

from nltk.corpus import wordnet

# Get synonyms of a word
phrase = 'the happy dog x ag ga er'
subwords = phrase.split(' ')

all_synonyms = []
# synonym expansion
for subword in subwords:
    for syn in wordnet.synsets(subword):
        for lemma in syn.lemmas():
            all_synonyms.append(lemma.name())

print(all_synonyms)

