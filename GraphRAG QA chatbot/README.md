# Enterprise GraphRAG QA Chatbot
This repository provides an GraphRAG Q&A Chatbot implementation
## Features

- Hybrid Retrieval (Vector + Graph)
- Neo4j Knowledge Graph
- ChromaDB Vector Store
- LangChain Integration
- Streamlit UI
- Entity-aware Q&A

## Setup

### Install Dependencies

```bash
pip install -r requirements.txt
```

### Install SpaCy Model

```bash
python -m spacy download en_core_web_sm
```

### Start Neo4j

```bash
docker run --name neo4j -p7474:7474 -p7687:7687 -d -e NEO4J_AUTH=neo4j/password neo4j:latest
```

### Build Graph

```bash
python graph_builder.py
```

### Build Vector DB

```python
from vector_store import VectorStore

vs = VectorStore()
vs.build_vector_store("data/documents")
```

### Run Chatbot

```bash
streamlit run app.py
```