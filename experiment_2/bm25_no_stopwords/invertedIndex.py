import math
import numpy as np
from typing import Dict
from index import Index
from documentVector import DocumentVector
from rank_bm25 import BM25Okapi


class InvertedIndexBM25:
    def __init__(self, index: Index) -> None:
        tokenized_corpus = index.get_tokenized_corpus()
        print("Tokenizing corpus")
        self.index = index
        self.bm25 = BM25Okapi(tokenized_corpus)
        print("Tokenizing corpus completed")
        pass

    def rank_query_in_bm25(self, query: str):
        tokenized_query = query.split(" ")
        scores = self.bm25.get_scores(tokenized_query)
        normalized_scores = (scores - np.min(scores)) / (np.max(scores) - np.min(scores))
        return normalized_scores.argsort()[::-1], normalized_scores


    def get_id_from_docs(self, key: str):
        return self.index.get_id_from_key(key=key)

    def get_doc_from_key(self, key: str):
        return self.index.get_joined_doc_corpus(key)


class InvertedIndex:
    def __init__(self, index: Index):
        self.inverted_index = {}
        self.doc_lengths_no_root = {}

        # For use later in a different class
        self.index = index

        # Building inverted_index map...
        # iterate documents, tokens -> populate inverted_index, corpusFrequency
        for doc_id in index.key_set():
            # iterate tokens associated to each document
            for token in index.key_set_for_doc(doc_id):
                token_frequency = index.frequency(doc_id, token)

                if token != "":
                    if token not in self.inverted_index:
                        self.inverted_index[token] = DocumentVector(
                            1, (math.log(79923 / 1) / math.log(2)), token_frequency
                        )
                        self.inverted_index[token].add_to_docs(doc_id, token_frequency)
                    else:
                        self.inverted_index[token].add_to_docs(doc_id, token_frequency)

                        # Updating df
                        self.inverted_index[token].set_df(
                            self.inverted_index[token].get_df() + 1
                        )

                        # Setting new max_tf if the document's tf is greater than the current max_tf
                        if token_frequency > self.inverted_index[token].get_max_tf():
                            self.inverted_index[token].set_max_tf(token_frequency)

                        # Updating idf
                        self.inverted_index[token].set_idf(
                            (
                                math.log(79923 / (self.inverted_index[token].get_df()))
                                / math.log(2)
                            )
                        )

        # Building doc_lengths_no_root map...
        for token in self.inverted_index.keys():
            for doc_no in self.inverted_index[token].get_docs().keys():
                tf_normalized = (
                    float(self.inverted_index[token].get_docs()[doc_no])
                    / self.inverted_index[token].get_max_tf()
                )
                idf = self.inverted_index[token].get_idf()
                if doc_no not in self.doc_lengths_no_root:
                    self.doc_lengths_no_root[doc_no] = math.pow(tf_normalized * idf, 2)
                else:
                    cur_length_no_root = self.doc_lengths_no_root[doc_no]
                    self.doc_lengths_no_root[doc_no] = cur_length_no_root + (
                        math.pow(tf_normalized * idf, 2)
                    )

    def get_max_tf(self, token: str) -> int:
        return self.inverted_index[token].get_max_tf()

    def contains_key(self, token: str) -> bool:
        return token in self.inverted_index

    def get_doc_length(self, doc_no: str) -> float:
        return math.sqrt(self.doc_lengths_no_root[doc_no])

    def get_token(self, token: str) -> DocumentVector:
        return self.inverted_index[token]

    def size(self) -> int:
        return len(self.inverted_index)

    def get_inverted_index(self) -> Dict[str, DocumentVector]:
        return self.inverted_index

    def get_index(self):
        return self.index

    def __str__(self) -> str:
        result = "Printing Inverted Index...\n"
        for key in self.inverted_index.keys():
            result += f"{key}: {self.inverted_index[key]}\n\n"

        result += "Printing Document Lengths...\n"
        for doc_no in self.doc_lengths_no_root.keys():
            result += f"{doc_no}: {self.get_doc_length(doc_no)}\n"

        result += "\n"
        return result
