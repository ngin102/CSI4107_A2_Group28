import os
try:
    from transformers import AutoModelForCausalLM, AutoTokenizer
except ImportError:
    print("Cannot find transformers module! Installing...")
    os.system("pip install transformers")
    from transformers import AutoModelForCausalLM, AutoTokenizer

try:
    import torch
except ImportError:
    print("Cannot find torch module! Installing...")
    os.system("pip install torch")
    import torch

try:
    from sklearn.metrics.pairwise import cosine_similarity
except ImportError:
    print("Cannot find cosine_similarity function! Installing...")
    os.system("pip install scikit-learn")
    from sklearn.metrics.pairwise import cosine_similarity

modelName = "EleutherAI/gpt-neo-1.3B"

class GPT3Retriever:
    def __init__(self) -> None:
        if torch.backends.mps.is_available():
            self.device = "mps"
            print("Using GPU")
        else:
            self.device = "cpu"
            print("Using CPU")

        self.model = AutoModelForCausalLM.from_pretrained(modelName, output_hidden_states=True).to(self.device)
        self.tokenizer = AutoTokenizer.from_pretrained(modelName)

        self.query = ""
        self.queryNo = -1
        self.queryInputIds = []

        self.documentIds = []
        self.documents = []
        self.documentEmbeddings = []

    def clearVariables(self):
        self.query = ""
        self.queryNo = -1
        self.queryInputIds = []

        self.documentIds = []
        self.documents = []
        self.documentEmbeddings = []

    def insertQuery(self, query: str, queryNo):
        self.query = query
        self.queryNo = queryNo

        self.queryInputIds = self.tokenizer.encode(self.query, return_tensors="pt").to(self.device)

    def insertDocument(self, doc: str, docNo: str):
        self.documents.append(doc)
        self.documentIds.append(docNo)

        max_length = self.tokenizer.model_max_length
        docInputIds = self.tokenizer.encode(doc, return_tensors="pt", truncation=True, max_length=max_length).to(self.device)
        
        with torch.no_grad():
            docHiddenStates = self.model(docInputIds).hidden_states[-2]
            docEmbedding = torch.mean(docHiddenStates, dim=1).squeeze()
            docEmbedding = docEmbedding / docEmbedding.norm()
        self.documentEmbeddings.append(docEmbedding)


    def retrieve(self):
        with torch.no_grad():
            queryHiddenStates = self.model(self.queryInputIds).hidden_states[-2]
            queryEmbedding = torch.mean(queryHiddenStates, dim=1).squeeze()
            
        print(f"queryEmbedding shape: {queryEmbedding.shape}")
        print(f"len(documentEmbeddings): {len(self.documentEmbeddings)}")

        if not self.documentEmbeddings:
            print("No documents found. Skipping retrieval.")
            return

        similarityScores = cosine_similarity(
            queryEmbedding.cpu().numpy().reshape(1, -1), torch.stack(self.documentEmbeddings).cpu().numpy()
        )

        ranked_docs = [
            (self.documentIds[i], score) for i, score in enumerate(similarityScores[0])
        ]
        ranked_docs.sort(key=lambda x: x[1], reverse=True)

        print(ranked_docs)

        file_name = "results_experiment1_GPTNeo.txt"
        count = 1
        with open(file_name, "a") as file:
            print(f"------------------------- 1000 for queryNo {self.queryNo} -------------------------")
            for idx in ranked_docs:
                rounded_value = "{:.4f}".format(idx[1])
                write_str = f"{self.queryNo} Q0 {idx[0]} {str(count)} {rounded_value} trial4"
                file.write(write_str)
                file.write(os.linesep)
                count += 1
