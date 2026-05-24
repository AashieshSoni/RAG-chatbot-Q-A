from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from document_processor import DocumentProcessor

class VectorStore:
    def __init__(self, persist_directory="./chroma_db"):
        self.persist_directory = persist_directory

        self.embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )

        self.processor = DocumentProcessor()

    def build_vector_store(self, docs_path):
        docs = self.processor.process_documents(docs_path)

        db = Chroma.from_documents(
            documents=docs,
            embedding=self.embeddings,
            persist_directory=self.persist_directory
        )

        db.persist()
        print("Vector DB created")

    def get_retriever(self):
        db = Chroma(
            persist_directory=self.persist_directory,
            embedding_function=self.embeddings
        )

        return db.as_retriever(search_kwargs={"k": 4})