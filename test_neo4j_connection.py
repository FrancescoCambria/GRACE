import os
from neo4j import GraphDatabase

# Manually parse .env to avoid dependency issues
env_path = '/home/cambria/MineGraphRule/GRAM/.env'
env_vars = {}
if os.path.exists(env_path):
    with open(env_path, 'r') as f:
        for line in f:
            if '=' in line and not line.startswith('#'):
                key, value = line.strip().split('=', 1)
                env_vars[key] = value

uri = env_vars.get('NEO4J_URI', 'neo4j://localhost:7687')
user = env_vars.get('NEO4J_USER', 'neo4j')
password = env_vars.get('NEO4J_PASSWORD', 'password')

print(f"Attempting to connect to: {uri}")

try:
    driver = GraphDatabase.driver(uri, auth=(user, password))
    with driver.session() as session:
        result = session.run("RETURN 1 AS one")
        record = result.single()
        if record and record["one"] == 1:
            print("SUCCESS: Connection to Neo4j is working.")
        else:
            print("FAILURE: Connection established but query failed.")
    driver.close()
except Exception as e:
    print(f"FAILURE: Could not connect to Neo4j. Error: {e}")
