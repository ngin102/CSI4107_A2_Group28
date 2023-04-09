import os
from invertedIndex import InvertedIndexBM25

class Retrieval:
    def __init__(self, ii: InvertedIndexBM25, printKey):
        self.invertedIndex = ii
        self.similarityScores = {}
        self.printKey = printKey

    def calculateSimilarity(self, query):
      
        idx_scores, scores = self.invertedIndex.rank_query_in_bm25(query=query)

        # Printing to file
        fileName = "results.txt"
        count = 1
        with open(fileName, "a") as file:
            for idx in idx_scores:
                roundedValue = "{:.4f}".format(scores[idx])
                key = self.invertedIndex.get_id_from_docs(idx)
                writeStr = (
                    self.printKey
                    + " Q0 "
                    + key
                    + " "
                    + str(count)
                    + " "
                    + roundedValue
                    + " base_bm25"
                )

                file.write(writeStr)
                count += 1

                if self.printKey != "50":
                    file.write(os.linesep)
                elif count != 1001 and self.printKey == "50":
                    file.write(os.linesep)

                if count == 1001:
                    break

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
