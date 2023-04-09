from invertedIndex import InvertedIndexBM25
from use_sbert import SBERT_Retrieval

class Retrieval:
    def __init__(self, ii: InvertedIndexBM25, printKey):
        self.invertedIndex = ii
        self.similarityScores = {}
        self.queryInfo = {}
        self.queryLength = 0.0
        self.printKey = printKey

    def calculateSimilarity(self, query, use_cls: SBERT_Retrieval):
        use_cls.insertQuery(self.printKey)

        count = 1
        for key in self.invertedIndex.get_docs():
            use_cls.insertDoc(self.invertedIndex.get_doc_from_key(key), key)
            print("Total documents encoded: " + str(count))
            count += 1

        use_cls.get_similarity()

    def __str__(self):
        thisString = ""

        print("Printing Similarity Scores...")
        count = 0
        for docNo in self.similarityScores:
            print(docNo + ": " + str(self.similarityScores[docNo]))
            count += 1

            if count == 1000:
                print("1000th document reached.")
                break

        print(str(len(self.similarityScores)) + " documents matched.")

        return thisString