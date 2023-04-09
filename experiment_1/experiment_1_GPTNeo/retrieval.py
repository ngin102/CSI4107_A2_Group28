import os
from invertedIndex import InvertedIndexBM25
from gpt3Retriever import GPT3Retriever


class Retrieval:
    def __init__(self, ii: InvertedIndexBM25, printKey):
        self.invertedIndex = ii
        self.similarityScores = {}
        self.queryInfo = {}
        self.queryLength = 0.0
        self.printKey = printKey

    def calculateSimilarity(self, query, use_cls: GPT3Retriever):

        idx_scores, scores = self.invertedIndex.rank_query_in_bm25(query=query)

        use_cls.insertQuery(query, self.printKey)

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
                    + " experiment1_GPTNeo"
                )
                
                file.write(writeStr)
                count += 1

                if count <= 1001:
                    use_cls.insertDocument(
                        self.invertedIndex.get_doc_from_key(key=key), key
                    )

                if self.printKey != "50":
                    file.write(os.linesep)
                elif count != 1001 and self.printKey == "50":
                    file.write(os.linesep)

                if count == 1001:
                    break

        use_cls.retrieve()

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
