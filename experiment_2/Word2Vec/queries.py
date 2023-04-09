import os
import re
from typing import Dict
from stemmer import PorterStemmer

try:
    import gensim.downloader as api
except ImportError:
    print("Can not find gensim library! Installing...")
    os.system("pip install gensim")
    import gensim.downloader as api


class Queries:
    def __init__(self):
        self.queries = {}
        self.stopwords = {}

        current_file_dir = os.path.dirname(os.path.abspath(__file__))
        file_path = os.path.join(current_file_dir, "topics1-50.txt")
        s = PorterStemmer()

        stopword_file = os.path.join(current_file_dir, "stopwords.txt")     

        # Getting stopwords; ideally do this with a map
        with open(stopword_file) as f:
            for subword in f.read().split():
                stemmed_word = s.stem(subword, 0, len(subword) - 1)
                self.stopwords[stemmed_word] = 1

        # Load Word2Vec model
        model = api.load('word2vec-google-news-300')

        if os.path.isfile(file_path):
            with open(file_path, 'r') as file:
                content = file.read()

                # Split the file into individual queries based on the <top> tag
                text_queries = content.split("<top>")
                for i in range(1, len(text_queries)):
                    query = text_queries[i]

                    query = re.sub("<num>", "", query)
                    query = re.sub("<narr>", "", query)
                    query = re.sub("<title>", "", query)
                    query = re.sub("<desc>", "", query)
                    query = re.sub("</top>", "", query)

                    query = query.lower() # convert all characters to lowercase
                    query = re.sub("[^a-zA-Z0-9\\s]", "", query) # Remove all punctuation and other special characters
                    query = re.sub("\\s+", " ", query) # Replace multiple spaces
                    query = re.sub("\\d", "", query) # Remove numbers
                    query = query.strip() # Remove trailing whitespaces

                    subwords = query.split(" ")

                    # Expand query with similar words
                    expanded_query = ""
                    for j in range(len(subwords)):
                        if subwords[j] not in self.stopwords:
                            word = subwords[j]
                            expanded_words = [word]

                            # Add similar words from Word2Vec model
                            if word in model:
                                similar_words = model.most_similar(word, topn=3) # Get top 3 similar words
                                for similar_word, similarity in similar_words:
                                    if similarity > 0.7: # Only include words with a high cosine similarity
                                        similar_word_processed = re.sub("_", " ", similar_word) # Convert _ to a space
                                        similar_word_processed = re.sub("[^a-zA-Z0-9\\s]", "", similar_word_processed) # Remove all punction
                                        similar_word_processed = re.sub("\\s+", " ", similar_word_processed) # Replace multiple spaces
                                        similar_word_processed = re.sub("\\d", "", similar_word_processed) # Remove numbers
                                        similar_word_processed = similar_word_processed.lower() # set to lowercase
                                        similar_word_processed = similar_word_processed.strip()
                                        if similar_word_processed not in expanded_words:
                                            expanded_words.append(similar_word_processed)

                            # Stem the expanded words
                            expanded_words = [s.stem(expanded_word, 0, len(expanded_word) - 1) for expanded_word in expanded_words]
                            expanded_query += " ".join(expanded_words) + " "

                    self.queries[str(i)] = expanded_query.strip()
                    #print(expanded_query + "\n\n")

    def get_queries(self) -> Dict[str, str]:
        return self.queries

    def __str__(self) -> str:
        result = ""
        for key in self.queries.keys():
            result += f"{key}: {self.queries[key]}\n"
        return result
