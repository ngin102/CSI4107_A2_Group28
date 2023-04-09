import os
import time
from preprocessor import Preprocessor
from index import Index
from invertedIndex import InvertedIndex, InvertedIndexBM25
from queries import Queries
from retrieval import Retrieval
from universalSentenceEncoder import UniversalSentenceEncoderRetrieval
from rank_bm25 import BM25Okapi


def main():
    current_file_dir = os.path.dirname(os.path.abspath(__file__))

    results_file = os.path.join(current_file_dir, "results.txt")
    results_use_experiment = os.path.join(
        current_file_dir, "results_experiment1_USE.txt"
    )
    if os.path.exists(results_file):
        print("Existing results.txt file detected. This file will be deleted.")
        print("A new results file will be written in its place.")
        os.remove(results_file)
        print(results_file, "has been deleted.\n")

    if os.path.exists(results_use_experiment):
        print("Existing results USE txt file detected. This file will be deleted.")
        print("A new file will be generated in it's place")
        os.remove(results_use_experiment)
        print(results_use_experiment, "has been deleted.\n")

    print("Attempting to read", os.path.join(current_file_dir, "coll"))
    preprocessor = Preprocessor(os.path.join(current_file_dir, "coll"))

    # ###########################################################################
    print("Reading documents; building index...")
    start = time.time()

    index = preprocessor.preprocess()
    print("Index built.")
    print("Number of documents read:", index.size(), ".")

    end = time.time()
    print((end - start), "seconds elapsed.\n")
    start = time.time()

    # ###########################################################################
    print("Building inverted index...")
    inverted_index = InvertedIndexBM25(index)
    print("Inverted index built. Using BM25")
    # print("Size of inverted index vocabulary:", inverted_index.size(), ".")
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

    use_retrieval = UniversalSentenceEncoderRetrieval()

    # ###########################################################################
    print("Retrieving cosine similarity scores for each query...")
    for key in queries.get_queries().keys():
        search = Retrieval(inverted_index, key)
        search.calculateSimilarity(
            queries.get_queries().get(key), use_cls=use_retrieval
        )
        use_retrieval.clearModel()
    print("Cosine similarity scores retrieved.")
    print("Please see results file (results.txt).")

    # ###########################################################################

    end = time.time()
    print((end - start), "seconds elapsed.\n")


if __name__ == "__main__":
    main()
