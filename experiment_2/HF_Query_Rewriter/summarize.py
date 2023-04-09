import pkg_resources

# List the packages you want to check
packages = ['gensim', 'transformers', 'torch']

# Check if each package is installed
for package in packages:
    try:
        pkg_resources.get_distribution(package)
        print(package, 'is already installed')
    except pkg_resources.DistributionNotFound:
        print(package, 'is not installed')
        if package == 'torch':
            import platform
            if platform.system() == 'Windows':
                subprocess.check_call(['pip', 'install', 'torch==1.9.0+cpu', '-f', 'https://download.pytorch.org/whl/torch_stable.html'])
            else:
                subprocess.check_call(['pip', 'install', 'torch'])
        else:
            # Use subprocess to run pip install command
            import subprocess
            subprocess.check_call(['pip', 'install', package])

from transformers import GPT2Tokenizer, GPT2Model, AutoTokenizer
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM



class Summarizer:
    def __init__(self):
        print("Initializing Summarizer")
        self.summarization_model = AutoModelForSeq2SeqLM.from_pretrained("facebook/bart-large-xsum")
        self.tokenizer = AutoTokenizer.from_pretrained("facebook/bart-large-xsum")
        print("Initialized Summarizer")

    def summarize(self, text):
        # initialize models for text summarization
        inputs = self.tokenizer(text, max_length=1024, return_tensors="pt", truncation=True)
        #summary_ids = self.summarization_model.generate(inputs["input_ids"], num_beams=2, max_length=len(text), early_stopping=True, min_length=len(text))
        summary_ids = self.summarization_model.generate(inputs["input_ids"], num_beams=1, max_length=len(text)*1.5, early_stopping=True)
        summary = self.tokenizer.decode(summary_ids[0], skip_special_tokens=True)
        print("Summarizing:")
        print(text[:len(summary)])
        print("TO->")
        print(summary)
        return summary

# summarizer = Summarizer()
# queries = ["Coping with overcrowded prisons","Accusations of Cheating by Contractors on U.S. Defense Projects","Insurance Coverage which pays for Long Term Care","Oil Spills","Right Wing Christian Fundamentalism in U.S."]
#
# for query in queries:
#     summarizer.summarize(query)

# class Summarizer:
#     def __init__(self):
#         print("Initializing Summarizer")
#         self.summarization_model = GPT2Model.from_pretrained('gpt2')
#         self.tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
#
#         print("Initialized Summarizer")
#
#     def summarize(self, text):
#         # initialize models for text summarization
#         inputs = self.tokenizer(text, max_length=1024, return_tensors="pt", truncation=True)
#         summary_ids = self.summarization_model.generate(inputs["input_ids"], num_beams=4, max_length=100, early_stopping=True)
#         summary = self.tokenizer.decode(summary_ids[0], skip_special_tokens=True)
#         print("Summarizing:")
#         print(text[:100])
#         print("TO->")
#         print(summary)
#         return summary