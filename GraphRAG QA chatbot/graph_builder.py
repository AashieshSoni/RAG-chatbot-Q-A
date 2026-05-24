import os
import spacy
from neo4j import GraphDatabase
from dotenv import load_dotenv
from document_processor import DocumentProcessor

load_dotenv()

class GraphBuilder:
    def __init__(self):
        self.nlp = spacy.load("en_core_web_sm")

        self.driver = GraphDatabase.driver(
            os.getenv("NEO4J_URI"),
            auth=(
                os.getenv("NEO4J_USERNAME"),
                os.getenv("NEO4J_PASSWORD")
            )
        )

        self.processor = DocumentProcessor()

    def close(self):
        self.driver.close()

    def extract_entities_relations(self, text):
        doc = self.nlp(text)

        entities = []
        relations = []

        for ent in doc.ents:
            entities.append({
                "name": ent.text,
                "label": ent.label_
            })

        for sent in doc.sents:
            sent_doc = self.nlp(sent.text)
            ents = [e.text for e in sent_doc.ents]

            if len(ents) >= 2:
                for i in range(len(ents)-1):
                    relations.append({
                        "source": ents[i],
                        "target": ents[i+1],
                        "relation": "RELATED_TO"
                    })

        return entities, relations

    def create_graph(self, documents_path):
        documents = self.processor.process_documents(documents_path)

        with self.driver.session() as session:
            for doc in documents:
                entities, relations = self.extract_entities_relations(
                    doc.page_content
                )

                for entity in entities:
                    session.run(
                        '''
                        MERGE (e:Entity {name: $name})
                        SET e.label = $label
                        ''',
                        name=entity["name"],
                        label=entity["label"]
                    )

                for rel in relations:
                    session.run(
                        '''
                        MATCH (a:Entity {name: $source})
                        MATCH (b:Entity {name: $target})
                        MERGE (a)-[:RELATED_TO]->(b)
                        ''',
                        source=rel["source"],
                        target=rel["target"]
                    )

        print("Knowledge graph created successfully")

if __name__ == "__main__":
    builder = GraphBuilder()
    builder.create_graph("data/documents")
    builder.close()