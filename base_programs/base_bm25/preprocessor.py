import os
import re
from copy import deepcopy

from index import Index
from stemmer import PorterStemmer


class Preprocessor:
    def __init__(self, input_folder):
        self.input_folder = input_folder
        self.index = Index()

    def preprocess(self):
        # Get a list of all the files in the input folder
        listOfFiles = os.listdir(self.input_folder)

        full_doc = {}

        s = PorterStemmer()

        for file_name in listOfFiles:
            file_path = os.path.join(self.input_folder, file_name)

            if os.path.isfile(file_path):
                try:
                    # Read the contents of the file
                    with open(file_path, "r") as file:
                        content = file.read()

                        # Split the file into individual documents based on the <DOC> tag
                        docs = content.split("<DOC>")
                        for i in range(1, len(docs)):
                            doc = docs[i]
                            doc_no = ""
                            p = re.compile("<DOCNO>(.*?)</DOCNO>")
                            m = p.search(doc)

                            if m:
                                doc_no = m.group(1).strip()

                            doc = p.sub("", doc)

                            p = re.compile("<FILEID>(.*?)</FILEID>", re.DOTALL)
                            m = p.search(doc)
                            doc = p.sub("", doc)

                            p = re.compile("<1ST_LINE>(.*?)</1ST_LINE>", re.DOTALL)
                            m = p.search(doc)
                            doc = p.sub("", doc)

                            p = re.compile("<2ND_LINE>(.*?)</2ND_LINE>", re.DOTALL)
                            m = p.search(doc)
                            doc = p.sub("", doc)

                            p = re.compile("<BYLINE>(.*?)</BYLINE>", re.DOTALL)
                            m = p.search(doc)
                            doc = p.sub("", doc)

                            p = re.compile("<DATELINE>(.*?)</DATELINE>", re.DOTALL)
                            m = p.search(doc)
                            doc = p.sub("", doc)

                            p = re.compile("<NOTE>(.*?)</NOTE>", re.DOTALL)
                            m = p.search(doc)
                            doc = p.sub("", doc)

                            p = re.compile("<HEAD>")
                            m = p.search(doc)
                            doc = p.sub("", doc)

                            p = re.compile("</HEAD>")
                            m = p.search(doc)
                            doc = p.sub("", doc)

                            p = re.compile("<TEXT>")
                            m = p.search(doc)
                            doc = p.sub("", doc)

                            p = re.compile("</TEXT>")
                            m = p.search(doc)
                            doc = p.sub("", doc)

                            p = re.compile("</DOC>")
                            m = p.search(doc)
                            doc = p.sub("", doc)

                            doc = doc.lower()  # convert all characters to lowercase
                            doc = re.sub(
                                r"[^a-zA-Z0-9\s]", "", doc
                            )  # Remove all punctuation and other special characters
                            doc = re.sub(r"\s+", " ", doc)  # Replace multiple spaces
                            doc = re.sub(r"\d", "", doc)  # Remove numbers
                            doc = doc.strip()  # Remove trailing whitespaces

                            full_doc[doc_no] = doc

                            subwords = doc.split()  # Split individual documents by word

                            stemmed_doc = ""

                            for subword in subwords:
                                stemmed_word = s.stem(subword, 0, len(subword) - 1)
                                # self.index.insert(stemmed_word, doc_no)
                                stemmed_doc += stemmed_word + " "

                            self.index.insert(stemmed_doc, doc_no)
                            print(
                                f"Finished processing {doc_no} - document count {len(list(full_doc.keys()))}"
                            )
                except IOError as e:
                    print(e)

        # Deepcopying the full doc map to index in case of any weird reference pointer python magic. worst case
        self.index.contain_ref_to_full_doc(full_doc=deepcopy(full_doc))
        return self.index

    def get_full_doc(self):
        return self.full_doc
