import math
import collections
import os
import itertools
import numpy
from invertedIndex import InvertedIndex, InvertedIndexBM25
from universalSentenceEncoder import UniversalSentenceEncoderRetrieval


class Retrieval:
    def __init__(self, ii: InvertedIndexBM25, printKey):
        self.invertedIndex = ii
        self.similarityScores = {}
        self.queryInfo = {}
        self.queryLength = 0.0
        self.printKey = printKey

    def calculateSimilarity(self, query, use_cls: UniversalSentenceEncoderRetrieval):
        # for searchWord in query.lower().split():
        #     if searchWord not in self.queryInfo:
        #         self.queryInfo[searchWord] = 1
        #     else:
        #         self.queryInfo[searchWord] += 1

        # for token in self.queryInfo:
        #     if self.invertedIndex.contains_key(token):
        #         tfq = float(self.queryInfo[token])
        #         idf = self.invertedIndex.get_token(token).get_idf()

        #         self.queryLength += math.pow((0.5 + (0.5 * tfq)) * idf, 2)

        #         for docNo in self.invertedIndex.get_token(token).get_docs():
        #             tfi_normalized = (
        #                 float(self.invertedIndex.get_token(token).get_docs()[docNo])
        #                 / self.invertedIndex.get_token(token).get_max_tf()
        #             )

        #             if docNo not in self.similarityScores:
        #                 self.similarityScores[docNo] = (tfi_normalized * idf) * (
        #                     (0.5 + (0.5 * tfq)) * idf
        #                 )
        #             else:
        #                 curSimilarityScore = self.similarityScores[docNo]
        #                 self.similarityScores[docNo] = curSimilarityScore + (
        #                     (tfi_normalized * idf) * ((0.5 + (0.5 * tfq)) * idf)
        #                 )

        # self.queryLength = math.sqrt(self.queryLength)

        # for docNo in self.similarityScores:
        #     currentSum = self.similarityScores[docNo]
        #     denominator = self.invertedIndex.get_doc_length(docNo) * self.queryLength
        #     self.similarityScores[docNo] = currentSum / denominator

        # # Sorting
        # sorted_map = collections.OrderedDict(
        #     sorted(self.similarityScores.items(), key=lambda x: x[1], reverse=True)
        # )
        # self.similarityScores = sorted_map

        idx_scores, scores = self.invertedIndex.rank_query_in_bm25(query=query)

        use_cls.insertQuery(query, self.printKey)

        # Printing to file
        fileName = "results.txt"
        count = 1
        with open(fileName, "a") as file:
            # for key in self.similarityScores:
            #     roundedValue = "{:.4f}".format(self.similarityScores[key])
            #     file.write(
            #         self.printKey
            #         + " Q0 "
            #         + key
            #         + " "
            #         + str(count)
            #         + " "
            #         + roundedValue
            #         + " trial4"
            #     )
            #     count += 1

            #     if count <= 1001:
            #         use_cls.insertDoc(
            #             self.invertedIndex.get_index().get_full_doc()[key], key
            #         )

            #     if self.printKey != "50":
            #         file.write(os.linesep)
            #     elif count != 1001 and self.printKey == "50":
            #         file.write(os.linesep)

            #     if count == 1001:
            #         break
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
                    + " trial4"
                )
                file.write(writeStr)
                count += 1

                if count <= 1001:
                    use_cls.insertDoc(self.invertedIndex.get_doc_from_key(key=key), key)

                if self.printKey != "50":
                    file.write(os.linesep)
                elif count != 1001 and self.printKey == "50":
                    file.write(os.linesep)

                if count == 1001:
                    break

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
