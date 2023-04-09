import os
import time
import pickle
from preprocessor import Preprocessor
from index import Index
from invertedIndex import InvertedIndex
from queries import Queries
from retrieval import Retrieval

def main():
    current_file_dir = os.path.dirname(os.path.abspath(__file__))

    results_file = os.path.join(current_file_dir, 'results.txt')
    if os.path.exists(results_file):
        print('Existing results.txt file detected. This file will be deleted.')
        print('A new results file will be written in its place.')
        os.remove(results_file)
        print(results_file, 'has been deleted.\n')
    
    print('Attempting to read',h.join(current_file_dir, 'coll'))
    preprocessor = Preprocessor(os.path.join(current_file_dir, 'coll'))
    
    # ###########################################################################
    print('Reading documents; building index...')
    start = time.time()
    
    index = preprocessor.preprocess()
    print('Index built.')
    print('Number of documents read:', index.size(), '.')
    
    end = time.time()
    print((end - start), 'seconds elapsed.\n')
    start = time.time()

    with open('index.pkl', 'wb') as f:
        pickle.dump(index, f)
    
    # ###########################################################################
    print('Building inverted index...')
    inverted_index = InvertedIndex(index)
    print('Inverted index built.')
    print('Size of inverted index vocabulary:', inverted_index.size(), '.')
    end = time.time()
    print((end - start), 'seconds elapsed.\n')
    start = time.time()

    with open('inverted_index.pkl', 'wb') as f:
        pickle.dump(inverted_index, f)
    
    # ###########################################################################
    print('Retrieving queries...')
    queries = Queries()
    print('Queries retrieved.')
    end = time.time()
    print((end - start), 'seconds elapsed.\n')
    start = time.time()
    
    # ###########################################################################
    print('Retrieving cosine similarity scores for each query...')
    for key in queries.get_queries().keys():
        search = Retrieval(inverted_index, key)
        search.calculateSimilarity(queries.get_queries().get(key))
    print('Cosine similarity scores retrieved.')
    print('Please see results file (results.txt).')
    
    end = time.time()
    print((end - start), 'seconds elapsed.\n')

if __name__ == '__main__':
    main()
