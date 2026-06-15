import os
from neo4j import GraphDatabase
from dotenv import load_dotenv

# Load credentials
load_dotenv("/home/cambria/MineGraphRule/GRAM/.env")

MEMGRAPH_URI = os.getenv("MEMGRAPH_URI", "bolt://localhost:7687")
MEMGRAPH_USER = os.getenv("MEMGRAPH_USER", "")
MEMGRAPH_PASSWORD = os.getenv("MEMGRAPH_PASSWORD", "")

class MemgraphConnector:
    def __init__(self, uri=MEMGRAPH_URI, user=MEMGRAPH_USER, password=MEMGRAPH_PASSWORD):
        self.uri = uri
        self.user = user
        self.password = password
        self.driver = None

    def connect(self):
        try:
            self.driver = GraphDatabase.driver(self.uri, auth=(self.user, self.password))
            print(f"Connected to Memgraph at {self.uri}")
        except Exception as e:
            print(f"Failed to connect to Memgraph: {e}")
            raise

    def close(self):
        if self.driver:
            self.driver.close()

    def run_query(self, query, parameters=None):
        if not self.driver:
            self.connect()
        with self.driver.session() as session:
            result = session.run(query, parameters)
            return [record for record in result]

    def get_law_triples(self):
        """
        Fetches triples relevant to the law dataset.
        """
        query = """
        MATCH (h)-[r]->(t)
        WHERE (h:Law OR h:Article OR h:Department OR h:Topic OR h:Government OR h:Legislature)
          AND (t:Law OR t:Article OR t:Department OR t:Topic OR t:Government OR t:Legislature)
        RETURN h.name, type(r), t.name
        """
        return self.run_query(query)

    def validate_rule(self, anchor_label, body_itemset, head_itemset):
        """
        Validates a rule against the live Memgraph graph.
        (Conceptual implementation of support/confidence check)
        """
        # This would involve constructing a Cypher query from the MGR syntax
        # and checking the counts in the database.
        pass

def main():
    connector = MemgraphConnector()
    try:
        connector.connect()
        print("Fetching a few law triples...")
        results = connector.run_query("MATCH (h:Law)-[r]->(t) RETURN h.name, type(r), t.name LIMIT 5")
        for record in results:
            print(f"{record[0]} -[{record[1]}]-> {record[2]}")
    finally:
        connector.close()

if __name__ == "__main__":
    main()
