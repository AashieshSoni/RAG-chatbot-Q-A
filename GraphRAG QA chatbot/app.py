import streamlit as st
from graph_rag_chain import GraphRAGChain

st.set_page_config(
    page_title="Enterprise GraphRAG QA Chatbot",
    layout="wide"
)

@st.cache_resource
def load_chain():
    return GraphRAGChain()

chain = load_chain()

st.title("Enterprise GraphRAG QA Chatbot")

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

question = st.chat_input("Ask a question")

if question:
    st.session_state.messages.append({
        "role": "user",
        "content": question
    })

    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            result = chain.query(question)

            st.markdown(result["answer"])

            with st.expander("Graph Context"):
                st.code(result["graph_context"])

            with st.expander("Vector Context"):
                st.code(result["vector_context"][:3000])

    st.session_state.messages.append({
        "role": "assistant",
        "content": result["answer"]
    })