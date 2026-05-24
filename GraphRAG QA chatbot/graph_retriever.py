import os
from neo4j import GraphDatabase
from dotenv import load_dotenv

load_dotenv()

class GraphRetriever:
    def __init__(self):
        self.driver = GraphDatabase.driver(
            os.getenv("NEO4J_URI"),
            auth=(
                os.getenv("NEO4J_USERNAME"),
                os.getenv("NEO4J_PASSWORD")
            )
        )

    def close(self):
        self.driver.close()

    def query_graph(self, question):
        keywords = question.split()

        cypher_query = '''
        MATCH (a)-[r]->(b)
        WHERE any(keyword IN $keywords
              WHERE toLower(a.name) CONTAINS toLower(keyword)
                 OR toLower(b.name) CONTAINS toLower(keyword))
        RETURN a.name as source,
               type(r) as relation,
               b.name as target
        LIMIT 10
        '''

        with self.driver.session() as session:
            result = session.run(
                cypher_query,
                keywords=keywords
            )

            data = []
            for row in result:
                data.append(
                    f"{row['source']} -[{row['relation']}]-> {row['target']}"
                )

        return "\n".join(data)