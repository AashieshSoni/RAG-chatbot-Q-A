import streamlit as st
from rag_chain import RAGChain
import os
from dotenv import load_dotenv

load_dotenv()

# Set page configuration
st.set_page_config(
    page_title="RAG Powered Q&A",
    page_icon="🤖",
    layout="wide"
)

# Initialize session state
if "rag_chain" not in st.session_state:
    st.session_state.rag_chain = RAGChain("my_documents")
if "messages" not in st.session_state:
    st.session_state.messages = []

# App title and description
st.title("🤖 RAG-Powered Q&A Chatbot")
st.markdown("""
This chatbot answers questions based on your docs using Retrieval-Augmented Generation (RAG).
""")

# Display chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat input
if prompt := st.chat_input("Ask a question about your documents"):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})

    # Display user message
    with st.chat_message("user"):
        st.markdown(prompt)

    # Get response from RAG chain
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            response = st.session_state.rag_chain.query(prompt)

        # Display assistant response
        st.markdown(response["result"])

        # Show source documents in an expander
        with st.expander("View source documents"):
            for i, doc in enumerate(response["source_documents"]):
                st.markdown(f"**Source {i+1}:**")
                st.markdown(f"Content: {doc.page_content[:500]}...")
                st.markdown(f"Metadata: {doc.metadata}")
                st.markdown("---")

    # Add assistant response to chat history
    st.session_state.messages.append(
        {"role": "assistant", "content": response["result"]})
