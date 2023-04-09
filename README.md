# CSI4107 - Assignment 2 (Group 28)

Adrian D'Souza (300066117), Nicholas Gin (300107597), Jared Wagner (300010832)

## Report
* **[(README) Our report](report.pdf)**: An overview and discussion of the work we did on this assignment and the results we obtained.

## Code (Experiments)
* **[Base programs (Assignment 1)](base_programs)**: Base versions of our IR System program that meet the criteria for Assignment 1.
  * **[BM25](base_programs/base_bm25)**: Base program that uses BM25 to rank documents against queries.
  * **[Cosine Similarity](base_programs/base_cosine)**: Base program that uses cosine similarity to rank documents against queries.
  
* **[Experiment 1](experiment_1)**: Versions of our IR System that we implemented for Experiment 1. 
  * **[BM25 + GPTNeo](experiment_1/experiment_1_GPTNeo)**: Version of our IR System that initially ranks documents using BM25 and then re-ranks them using a GPTNeo model.
  * **[BM25 + SBERT](experiment_1/experiment_1_SBERT)**: Version of our IR System that initially ranks documents using BM25 and then re-ranks them using a SBERT model.
  * **[BM25 + USE](experiment_1/experiment_1_USE)**: Version of our IR System that initially ranks documents using BM25 and then re-ranks them using a USE model.

* **[Experiment 2](experiment_2)**: Versions of our IR System that we implemented for Experiment 2. 
  * **[Word2Vec](experiment_2/Word2Vec)**: Version of our IR System that discriminately adds related words to a query.
  * **[Hugging Face Query Rewriter](experiment_2/HF_Query_Rewriter)**: Version of our IR System that re-writes queries using a Hugging Face text summarization model.
  * **[NLTK Query Synonymizer](experiment_2/NLTK_Query_Synonymizer)**: Version of our IR System that expands a query by indisciminately adding synonyms for each query token word.
  * **[Token Frequency Exclusion](experiment_2/NLTK_Query_Synonymizer/index.py)**: Commented out here. To use this requires commenting def get_tokenized_corpus(self) and uncommenting def get_tokenized_corpus(self, freq_threshold=10) with the desired frequency inclusion threshhold. This is used in the "ad-hoc" experiment at the end of the report.

* **[Experiment 3](experiment_3)**: Versions of our IR System that we implemented for Experiment 3. 
  * **[SBERT](experiment_3/experiment_3_SBERT)**: Version of our IR System that ranks documents using a SBERT model.
  * **[sent2vec](experiment_3/experiment_3_sent2vec)**: Version of our IR System that ranks documents using a sent2vec model.
  
