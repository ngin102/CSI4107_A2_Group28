import os
from typing import Dict

try:
    import numpy as np
except ImportError:
    print("Can not find numpy library! Installing...")
    os.system("pip install numpy")
    import numpy as np

try:
    import pickle
except ImportError:
    print("Can not find pickle library! Installing...")
    os.system("pip install pickle-mixin")
    import pickle

try:
    import gensim
   # from gensim.models import KeyedVectors
except ImportError:
    print("Can not find gensim library! Installing...")
    os.system("pip install gensim")
    import gensim
   # from gensim.models import KeyedVectors

try:
    import faiss
except ImportError:
    print("Can not find faiss-cpu library! Installing...")
    os.system("pip install faiss-cpu")
    import faiss

try:
    import sent2vec
except ImportError:
    print("Can not find sent2vec library! Installing...")
    os.system("pip install cython")
    os.system("git clone https://github.com/epfml/sent2vec.git")
    os.system("cd sent2ve")
    os.system("python setup.py build_ext --inplace")
    os.system("pip install .")
    import sent2vec


class Retrieval:
    def __init__(self, ii, oi, printKey):
        self.invertedIndex = ii
        self.index = oi
        self.similarityScores = {}
        self.queryInfo = {}
        self.queryLength = 0.0
        self.printKey = printKey
        self.embeddings_computed = False
        self.embeddings = None
        self.faiss_index = None

        current_file_dir = os.path.dirname(os.path.abspath(__file__))
        self.embeddings_path = os.path.join(current_file_dir, "embeddings.pkl")

        if os.path.exists(self.embeddings_path):
            self.embeddings_computed = True
            print("Embeddings already computed.")
            with open(self.embeddings_path, 'rb') as f:
                self.embeddings = pickle.load(f)
                print("Embeddings loaded.")

    @staticmethod
    def get_embedding(text, model):
        tokens = gensim.utils.simple_preprocess(text)
        return model.embed_sentence(" ".join(tokens))

    def calculateSimilarity(self, query):
        current_file_dir = os.path.dirname(os.path.abspath(__file__))
        model_path = os.path.join(current_file_dir, "wiki_unigrams.bin")
        sent2vec_model = sent2vec.Sent2vecModel()
        sent2vec_model.load_model(model_path)
        print("Loaded model.")

        if not self.embeddings_computed:
            doc_ids = list(self.index.get_entire_docs().keys())
            doc_texts = list(self.index.get_entire_docs().values())
            self.embeddings = np.vstack([Retrieval.get_embedding(text, sent2vec_model) for text in doc_texts])
            with open(self.embeddings_path, 'wb') as f:
                pickle.dump(self.embeddings, f)
                print("Saved embeddings.")

        embedding_dim = self.embeddings.shape[1]

        # Use cosine similarity directly by creating an IndexFlatIP index
        self.faiss_index = faiss.IndexFlatIP(embedding_dim)

        # Normalize the embeddings to ensure that dot product will give cosine similarity
        faiss.normalize_L2(self.embeddings)
        self.faiss_index.add(self.embeddings)
        self.embeddings_computed = True

        query_embedding = Retrieval.get_embedding(query, sent2vec_model)

        # Normalize the query embedding to ensure that dot product will give cosine similarity
        faiss.normalize_L2(query_embedding)

        # Search for the 1000 most similar documents using cosine similarity
        D, I = self.faiss_index.search(query_embedding.reshape(1, -1), k=1000)

        # Sort the results by descending cosine similarity
        sorted_indexes = I[0]
        sorted_scores = D[0]

        doc_ids_list = list(self.index.get_entire_docs().keys())
        top_1000_similar_strings = [(doc_ids_list[idx], score) for idx, score in zip(sorted_indexes, sorted_scores)]
        
        fileName = "results_experiment3_sent2vec.txt"
        count = 1
        with open(fileName, 'a') as file:
            for (doc_id, score) in top_1000_similar_strings:
                roundedValue = "{:.4f}".format(score)
                file.write(self.printKey + " Q0 " + doc_id + " " + str(count) + " " + roundedValue + " experiment3_sent2vec")
                count += 1

                if self.printKey != "50":
                    file.write(os.linesep)
                elif count != 1001 and self.printKey == "50":
                    file.write(os.linesep)

                if count == 1001:
                    break