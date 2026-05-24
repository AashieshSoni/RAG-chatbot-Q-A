import os
from langchain.document_loaders import PyPDFLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings


class DocumentProcessor:
    def __init__(self, chunk_size=1000, chunk_overlap=200):
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
        )
        # Use SentenceTransformers for embeddings
        self.embeddings = HuggingFaceEmbeddings(
            model_name="all-MiniLM-L6-v2",
            model_kwargs={'device': 'cuda'},  # Use CUDA if available
            encode_kwargs={'normalize_embeddings': False}
        )

    def load_documents(self, directory_path):
        documents = []
        for filename in os.listdir(directory_path):
            file_path = os.path.join(directory_path, filename)
            if filename.endswith('.pdf'):
                loader = PyPDFLoader(file_path)
                documents.extend(loader.load())
            elif filename.endswith('.txt'):
                loader = TextLoader(file_path)
                documents.extend(loader.load())
        return documents

    def process_documents(self, directory_path):
        # Load documents
        raw_documents = self.load_documents(directory_path)
        print(f"Loaded {len(raw_documents)} documents")

        # Split documents
        documents = self.text_splitter.split_documents(raw_documents)
        print(f"Split into {len(documents)} chunks")

        return documents
