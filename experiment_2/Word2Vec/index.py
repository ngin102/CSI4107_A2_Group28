import os
import re
from stemmer import PorterStemmer


class Index:
    def __init__(self):
        self.index = {}
        self.stopwords = {}
        self.numDocs = 0
        self.full_doc = {}

        s = PorterStemmer()
        current_file_dir = os.path.dirname(os.path.abspath(__file__))
        stopword_file = os.path.join(current_file_dir, "stopwords.txt")

        # Getting stopwords; ideally do this with a map
        with open(stopword_file) as f:
            for subword in f.read().split():
                stemmed_word = s.stem(subword, 0, len(subword) - 1)
                self.stopwords[stemmed_word] = 1

    def insert(self, token, document_id):
        if document_id not in self.index:
            self.index[document_id] = token.split(" ")
            self.numDocs += 1
            pass
        # if document_id not in self.index and token not in self.stopwords:
        #     self.index[document_id] = {token: 1}
        #     self.numDocs += 1
        # else:
        #     if token not in self.stopwords:
        #         if token not in self.index[document_id]:
        #             self.index[document_id][token] = 1
        #             self.numDocs += 1
        #         else:
        #             self.index[document_id][token] += 1
        pass

    def contain_ref_to_full_doc(self, full_doc):
        self.full_doc = full_doc

    def get_full_doc(self):
        return self.full_doc

    def frequency(self, doc_id, token):
        return self.index[doc_id][token]

    def get_id_from_key(self, key: str):
        return list(self.key_set())[key]

    def get_tokenized_corpus(self):
        return [self._get_doc_in_corpus(doc_id) for doc_id in self.index.keys()]

    def get_joined_doc_corpus(self, doc_id):
        return self._get_full_doc(doc_id)

    def _get_doc_in_corpus(self, doc_id):
        return self.index[doc_id]

    def _get_full_doc(self, doc_id):
        return self.get_full_doc()[doc_id]

    def key_set(self):
        return self.index.keys()

    def key_set_for_doc(self, doc_id):
        return self.index[doc_id].keys()

    def size(self):
        return self.numDocs

    def __str__(self):
        return str(self.index)