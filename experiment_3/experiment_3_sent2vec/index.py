import os
import re
from stemmer import PorterStemmer

class Index:
    def __init__(self):
        self.index = {}
        self.full_docs = {}
        self.stopwords = {}
        self.numDocs = 0

        s = PorterStemmer()
        current_file_dir = os.path.dirname(os.path.abspath(__file__))
        stopword_file = os.path.join(current_file_dir, "stopwords.txt")     

        # Getting stopwords; ideally do this with a map
        with open(stopword_file) as f:
            for subword in f.read().split():
                stemmed_word = s.stem(subword, 0, len(subword) - 1)
                self.stopwords[stemmed_word] = 1

    def insert(self, token, document_id):
        if document_id not in self.index and token not in self.stopwords:
            self.index[document_id] = {token: 1}
            self.numDocs += 1
        else:
            if token not in self.stopwords:
                if token not in self.index[document_id]:
                    self.index[document_id][token] = 1
                else:
                    self.index[document_id][token] += 1
    
    def insert_full_doc(self, full_doc, doc_no):
        self.full_docs[doc_no] = full_doc

    def get_entire_docs(self):
        return self.full_docs

    def frequency(self, doc_id, token):
        return self.index[doc_id][token]

    def key_set(self):
        return self.index.keys()

    def key_set_for_doc(self, doc_id):
        return self.index[doc_id].keys()

    def size(self):
        return self.numDocs, len(self.full_docs)

    def __str__(self):
        return str(self.index)
