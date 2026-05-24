from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate

from vector_store import VectorStore
from graph_retriever import GraphRetriever

load_dotenv()

class GraphRAGChain:
    def __init__(self):
        self.vector_store = VectorStore()
        self.retriever = self.vector_store.get_retriever()

        self.graph_retriever = GraphRetriever()

        self.llm = ChatOpenAI(
            model="gpt-4o-mini",
            temperature=0
        )

        self.prompt_template = '''
You are an enterprise GraphRAG assistant.

Use BOTH:
1. Vector document context
2. Knowledge graph relationships

to answer the question.

Vector Context:
{context}

Knowledge Graph Context:
{graph_context}

Question:
{question}

Answer:
'''

        self.prompt = PromptTemplate(
            template=self.prompt_template,
            input_variables=[
                "context",
                "graph_context",
                "question"
            ]
        )

    def query(self, question):
        docs = self.retriever.get_relevant_documents(question)

        vector_context = "\n\n".join([
            d.page_content for d in docs
        ])

        graph_context = self.graph_retriever.query_graph(question)

        final_prompt = self.prompt.format(
            context=vector_context,
            graph_context=graph_context,
            question=question
        )

        response = self.llm.invoke(final_prompt)

        return {
            "answer": response.content,
            "vector_context": vector_context,
            "graph_context": graph_context
        }