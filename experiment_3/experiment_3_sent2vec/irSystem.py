import os
import time
try:
    import pickle
except ImportError:
    print("Cannot find pickle! Installing...")
    os.system("pip install pickle-mixin")
    import pickle

from preprocessor import Preprocessor
from invertedIndex import InvertedIndex
from queries import Queries
from retrieval import Retrieval


def main():
    current_file_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(current_file_dir, "wiki_unigrams.bin")
    if not os.path.exists(model_path):
        print("sent2vec model not downloaded.")
        print("Please download the model from https://drive.google.com/file/d/0B6VhzidiLvjSa19uYWlLUEkzX3c/view?resourcekey=0-p9iI_hJbCuNiUq5gWz7Qpg to proceed.")
        print("Place the model in the directory with the rest of the files for this program.")
        return

    results_file = os.path.join(current_file_dir, "results.txt")
    results_use_experiment = os.path.join(
        current_file_dir, "results_experiment3_sent2vec.txt"
    )
    if os.path.exists(results_file):
        print("Existing results.txt file detected. This file will be deleted.")
        print("A new results file will be written in its place.")
        os.remove(results_file)
        print(results_file, "has been deleted.\n")

    if os.path.exists(results_use_experiment):
        print("Existing results sent2vec txt file detected. This file will be deleted.")
        print("A new file will be generated in its place.")
        os.remove(results_use_experiment)
        print(results_use_experiment, "has been deleted.\n")
    
    index_file = os.path.join(current_file_dir, 'index.pkl')
    if os.path.exists(index_file):
        print('Existing index.pkl file detected.')
        start = time.time()

        print('Loading index...')
        with open('index.pkl', 'rb') as f:
            index = pickle.load(f)    

        print('Index loaded.')
        print("Number of documents read:", index.size())

        end = time.time()
        print((end - start), "seconds elapsed.\n")
    else:
        print('No existing index.pkl file detected.')
        print("Attempting to read", os.path.join(current_file_dir, "coll"))
        preprocessor = Preprocessor(os.path.join(current_file_dir, "coll"))

        # ###########################################################################
        print("Reading documents; building index...")
        start = time.time()

        index = preprocessor.preprocess()
        print("Index built.")

        with open(index_file, 'wb') as f:
            pickle.dump(index, f)
            print("Saved index.pkl")

        print("Number of documents read:", index.size())

        end = time.time()
        print((end - start), "seconds elapsed.\n")
    
    #############################################################################
    inverted_index_file = os.path.join(current_file_dir, 'inverted_index.pkl')
    if os.path.exists(inverted_index_file):
        print('Existing inverted_index.pkl file detected.')
        start = time.time()

        print('Loading inverted index...')
        with open('inverted_index.pkl', 'rb') as f:
            inverted_index = pickle.load(f)    
        print('Inverted index loaded.')
        
        end = time.time()
        print((end - start), "seconds elapsed.\n")

    else:
        print('No existing inverted_index.pkl file detected.')
        start = time.time()
        print("Building inverted index...")
        inverted_index = InvertedIndex(index)
        print("Inverted index built.")

        with open(inverted_index_file, 'wb') as f:
            pickle.dump(inverted_index, f)
            print("Saved inverted_index.pkl")
            
        end = time.time()
        print((end - start), "seconds elapsed.\n")
    
    start = time.time()

    # ###########################################################################
    print("Retrieving queries...")
    queries = Queries()
    print("Queries retrieved.")
    end = time.time()
    print((end - start), "seconds elapsed.\n")
    start = time.time()

    # ###########################################################################
    print("Retrieving cosine similarity scores for each query...")
    for key in queries.get_queries().keys():
        search = Retrieval(inverted_index, index, key)
        search.calculateSimilarity(
            queries.get_queries().get(key)
        )
    print("Cosine similarity scores retrieved.")
    print("Please see results file (results_sent2vec.txt).")

    # ###########################################################################

    end = time.time()
    print((end - start), "seconds elapsed.\n")


if __name__ == "__main__":
    main()
