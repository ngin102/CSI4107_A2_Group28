class DocumentVector:
    """
    Represents the info about a term (meant to be used for the inverted index)
    """
    def __init__(self, df, idf, max_tf):
        self.docs = {}  # List of documents that the term appears in with
        self.df = df  # Document frequency of the term
        self.idf = idf  # Calculated idf value
        self.max_tf = max_tf  # The highest frequency of the term for any document it appears in

    def add_to_docs(self, doc_no, tf):
        self.docs[doc_no] = tf

    def set_df(self, new_df):
        self.df = new_df

    def set_idf(self, new_idf):
        self.idf = new_idf

    def set_max_tf(self, new_max_tf):
        self.max_tf = new_max_tf

    def get_docs(self):
        return self.docs

    def get_df(self):
        return self.df

    def get_idf(self):
        return self.idf

    def get_max_tf(self):
        return self.max_tf

    def __str__(self):
        this_string = ""

        print("df = " + str(self.df) + ", idf = " + str(self.idf) + ", max_tf = " + str(self.max_tf) + ", docs = ", end="")

        for doc_no, tf in self.docs.items():
            print("{" + doc_no + ": " + str(tf) + "} ", end="")

        return this_string
