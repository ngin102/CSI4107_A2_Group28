import os

try:
    import tensorflow_hub as hub
except ImportError:
    print("Cannot find tensorflow_hub! Installing...")
    os.system("pip install tensorflow_hub")
    import tensorflow_hub as hub

try:
    import numpy as np
except ImportError:
    print("Cannot find numpy! Installing...")
    os.system("pip install numpy")
    import numpy as np


class UniversalSentenceEncoderRetrieval:
    def __init__(self) -> None:
        print("Installing model for Universal Sentence Encoder from google tfhub.dev")
        print(
            "Warning: If not previously installed, it will take a couple minutes before the program continues"
        )
        self.use_model = hub.load(
            "https://tfhub.dev/google/universal-sentence-encoder/4"
        )
        # self.full_queries = full_queries
        self.documentEmbeddings = []
        self.queryEmbeddings = []
        self.documents = []
        self.documentIds = []
        self.queryNo = -1
        self.query = {}
        pass

    def clearModel(self):
        self.query = ""
        self.queryNo = -1
        self.queryEmbeddings = []
        self.documentEmbeddings = []
        self.documentIds = []
        self.documents = []

    def insertQuery(self, query, queryNo):
        self.query = query
        self.queryNo = queryNo
        self.queryEmbeddings = self.use_model([self.query])
        pass

    def insertDoc(self, doc, docNo):
        self.documents.append(doc)
        self.documentIds.append(docNo)

        docEmbeds = self.use_model([doc])
        self.documentEmbeddings.append(docEmbeds)
        pass

    def get_similarity(self):
        self.documentEmbeddings = np.concatenate(self.documentEmbeddings, axis=0)
        cosine_similarities = np.dot(
            self.queryEmbeddings, np.transpose(self.documentEmbeddings)
        )

        idxs = cosine_similarities[0].argsort()[::-1]

        fileName = "results_experiment1_USE.txt"
        count = 1
        with open(fileName, "a") as file:
            print(
                f"------------------------- 1000 for queryNo {self.queryNo} -------------------------"
            )
            for idx in idxs:
                roundedValue = "{:.4f}".format(cosine_similarities[0][idx])
                writeStr = f"{self.queryNo} Q0 {self.documentIds[idx]} {str(count)} {roundedValue} trial4"
                # print(writeStr)

                file.write(writeStr)
                file.write(os.linesep)
                count += 1
            # print(
            #     f"------------------------- 1000 for queryNo {self.queryNo} -------------------------"
            # )
        pass
