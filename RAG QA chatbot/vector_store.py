import chromadb
from chromadb.config import Settings
from document_processor import DocumentProcessor
import uuid
import os
import pathlab

class VectorStore:
    def __init__(self, persist_directory="./chroma_db"):
        self.client = chromadb.Client(Settings(
            chroma_db_impl="duckdb+parquet",
            persist_directory=persist_directory
        ))
        self.persist_directory = persist_directory
        self.document_processor = DocumentProcessor()

    def create_collection(self, collection_name):
        # Check if collection exists, create if not
        try:
            collection = self.client.get_collection(collection_name)
            print(f"Collection {collection_name} already exists")
            return collection
        except:
            collection = self.client.create_collection(collection_name)
            print(f"Created new collection: {collection_name}")
            return collection

    def add_documents(self, collection_name, directory_path):
        # Process documents
        documents = self.document_processor.process_documents(directory_path)

        # Get collection
        collection = self.create_collection(collection_name)

        # Prepare data for ChromaDB
        ids = [str(uuid.uuid4()) for _ in range(len(documents))]
        texts = [doc.page_content for doc in documents]
        metadatas = [doc.metadata for doc in documents]

        # Add to collection
        collection.add(
            ids=ids,
            documents=texts,
            metadatas=metadatas
        )

        # Persist the database
        self.client.persist()

        print(
            f"Added {len(documents)} documents to collection {collection_name}")
        return len(documents)

    def query_collection(self, collection_name, query_text, n_results=3):
        collection = self.client.get_collection(collection_name)
        results = collection.query(
            query_texts=[query_text],
            n_results=n_results
        )
        return results


# Example usage
if __name__ == "__main__":
    vs = VectorStore()
    # Put your PDFs/txt files in ./documents
    vs.add_documents("my_documents", "./documents")
