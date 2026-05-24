from langchain.chains import RetrievalQA
from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
from vector_store import VectorStore
import os
from dotenv import load_dotenv

load_dotenv()


class RAGChain:
    def __init__(self, collection_name):
        self.vector_store = VectorStore()
        self.collection = self.vector_store.client.get_collection(
            collection_name)
        self.llm = OpenAI(
            temperature=0,
            openai_api_key=os.getenv("OPENAI_API_KEY")
        )

        # Create a custom prompt
        self.prompt_template = """Use the following pieces of context to answer the question at the end. If you don't know the answer, just say that you don't know, don't try to make up an answer.

        {context}

        Question: {question}
        Answer:"""

        self.PROMPT = PromptTemplate(
            template=self.prompt_template,
            input_variables=["context", "question"]
        )

    def get_retriever(self):
        # Create a retriever from the Chroma collection
        from langchain.vectorstores import Chroma
        from langchain.embeddings import HuggingFaceEmbeddings

        embeddings = HuggingFaceEmbeddings(
            model_name="all-MiniLM-L6-v2",
            model_kwargs={'device': 'cuda'},
            encode_kwargs={'normalize_embeddings': False}
        )

        langchain_chroma = Chroma(
            client=self.vector_store.client,
            collection_name=self.collection.name,
            embedding_function=embeddings
        )

        return langchain_chroma.as_retriever(search_kwargs={"k": 3})

    def create_chain(self):
        chain_type_kwargs = {"prompt": self.PROMPT}
        retriever = self.get_retriever()

        qa_chain = RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type="stuff",
            retriever=retriever,
            chain_type_kwargs=chain_type_kwargs,
            return_source_documents=True
        )

        return qa_chain

    def query(self, question):
        qa_chain = self.create_chain()
        result = qa_chain({"query": question})
        return result


# Example usage
if __name__ == "__main__":
    rag = RAGChain("my_documents")
    result = rag.query("What is this document about?")
    print("Answer:", result["result"])
    print("Source documents:", result["source_documents"])
