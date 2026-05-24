from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader, TextLoader
import os

class DocumentProcessor:
    def __init__(self):
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )

    def load_documents(self, docs_path):
        docs = []

        for file in os.listdir(docs_path):
            path = os.path.join(docs_path, file)

            if file.endswith(".pdf"):
                loader = PyPDFLoader(path)
                docs.extend(loader.load())

            elif file.endswith(".txt"):
                loader = TextLoader(path)
                docs.extend(loader.load())

        return docs

    def process_documents(self, docs_path):
        docs = self.load_documents(docs_path)
        return self.splitter.split_documents(docs)