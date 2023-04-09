import os

try:
    import pickle
except ImportError:
    print("Cannot find pickle! Installing...")
    os.system("pip install pickle-mixin")
    import pickle

try:
    import numpy as np
except ImportError:
    print("Cannot find numpy! Installing...")
    os.system("pip install numpy")
    import numpy as np

try:
    import torch
    from sentence_transformers import SentenceTransformer
except ImportError:
    print("Cannot find sentence_transformers or torch! Installing...")
    os.system("pip install sentence-transformers torch")
    import torch
    from sentence_transformers import SentenceTransformer

try:
    from sklearn.metrics.pairwise import cosine_similarity
except ImportError:
    print("Cannot find scikit-learn! Installing...")
    os.system("pip install scikit-learn")
    from sklearn.metrics.pairwise import cosine_similarity


class SBERT_Retrieval:
    def __init__(self, full_queries) -> None:
        # Check if GPU is available and set device
        if torch.backends.mps.is_available():
            self.device = "mps" 
            print("Using GPU")
        else:
            self.device = "cpu"
            print("Using CPU")
            
        # Load the model on the available device
        self.model = SentenceTransformer('paraphrase-distilroberta-base-v2', device=self.device)
        self.full_queries = full_queries
        self.documentEmbeddings = []
        self.queryEmbeddings = []
        self.documents = []
        self.documentIds = []
        self.queryNo = -1
        self.query = {}

    def clearModel(self):
        self.query = ""
        self.queryNo = -1
        self.queryEmbeddings = []
        self.documentEmbeddings = []
        self.documentIds = []
        self.documents = []

    def insertQuery(self, queryNo):
        self.query = self.full_queries[queryNo]
        self.queryNo = queryNo
        self.queryEmbeddings = self.model.encode(self.query)

    def insertDoc(self, doc, docNo):
        self.documents.append(doc)
        self.documentIds.append(docNo)

        docEmbeds = self.model.encode(doc)
        self.documentEmbeddings.append(docEmbeds)

    def get_similarity(self):
        # Move embeddings to device (GPU or CPU)
        query_embeddings = torch.tensor(
            np.array(self.queryEmbeddings), device=self.device
            )
        doc_embeddings = torch.tensor(
            np.array(self.documentEmbeddings), device=self.device
            )

        try:
            current_file_dir = os.path.dirname(os.path.abspath(__file__))
            query_path = "embeddings/query_embeddings" + str(self.queryNo) + ".pkl"
            query_file = os.path.join(current_file_dir, query_path)
            with open(query_file, 'wb') as f:
                pickle.dump(query_embeddings, f)
                message = "Saved " + query_path
                print(message)
        except Exception as e:
            print("Could not save:", str(e))

        try:
            current_file_dir = os.path.dirname(os.path.abspath(__file__))
            doc_path = "embeddings/doc_embeddings" + str(self.queryNo) + ".pkl"
            doc_file = os.path.join(current_file_dir, doc_path)
            with open(doc_file, 'wb') as f:
                pickle.dump(doc_embeddings, f)
                message = "Saved " + doc_path
                print(message)
        except Exception as e:
            print("Could not save:", str(e))

        # Reshape embeddings to 2D arrays
        query_embeddings = query_embeddings.reshape(1, -1)
        doc_embeddings = doc_embeddings.reshape(len(self.documentIds), -1)

        cosine_similarities = cosine_similarity(query_embeddings.cpu().detach().numpy(), doc_embeddings.cpu().detach().numpy()).squeeze()
        normalized_scores = (cosine_similarities + 1) / 2  # Normalize scores to [0, 1]
        idxs = np.argsort(normalized_scores)[::-1]

        file_name = "results_experiment1_SBERT.txt"
        count = 1
        with open(file_name, "a") as file:
            print(f"------------------------- 1000 for queryNo {self.queryNo} -------------------------")
            for idx in idxs:
                rounded_value = "{:.4f}".format(normalized_scores[idx])
                write_str = f"{self.queryNo} Q0 {self.documentIds[idx]} {str(count)} {rounded_value} experiment1_SBERT"
                file.write(write_str)
                file.write(os.linesep)
                count += 1
