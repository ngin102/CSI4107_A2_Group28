import os
import time
try:
    import pickle
except ImportError:
    print("Cannot find pickle! Installing...")
    os.system("pip install pickle-mixin")
    import pickle

from preprocessor import Preprocessor
from invertedIndex import InvertedIndexBM25
from queries import Queries
from retrieval import Retrieval
from use_sbert import SBERT_Retrieval


def main():
    current_file_dir = os.path.dirname(os.path.abspath(__file__))

    results_file = os.path.join(current_file_dir, "results.txt")
    results_use_experiment = os.path.join(
        current_file_dir, "results_experiment1_SBERT.txt"
    )
    if os.path.exists(results_file):
        print("Existing results.txt file detected. This file will be deleted.")
        print("A new results file will be written in its place.")
        os.remove(results_file)
        print(results_file, "has been deleted.\n")

    if os.path.exists(results_use_experiment):
        print("Existing results SBERT txt file detected. This file will be deleted.")
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
        inverted_index = InvertedIndexBM25(index)
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

    sbert_retrieval = SBERT_Retrieval(queries.get_full_queries())

    # ###########################################################################
    print("Retrieving cosine similarity scores for each query...")
    for key in queries.get_queries().keys():
        print(key)
        search = Retrieval(inverted_index, key)
        search.calculateSimilarity(
            queries.get_queries().get(key), use_cls=sbert_retrieval
        )
        sbert_retrieval.clearModel()
    print("Cosine similarity scores retrieved.")
    print("Please see results file (results.txt).")

    # ###########################################################################

    end = time.time()
    print((end - start), "seconds elapsed.\n")


if __name__ == "__main__":
    main()
