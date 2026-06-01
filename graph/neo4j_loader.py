from neo4j import GraphDatabase

class Neo4jLoader:
    def __init__(self, uri, user, password):
        self.driver = GraphDatabase.driver(uri, auth=(user, password))

    def close(self):
        self.driver.close()

    def load(self, graph_records):
        with self.driver.session() as s:
            for g in graph_records:
                for node in g.nodes:
                    s.run(
                        f"""
                        MERGE (n:{node.node_type} {{id: $id}})
                        SET n += $props
                        """,
                        id=node.node_id,
                        props=node.properties
                    )

                for e in g.edges:
                    s.run(
                        f"""
                        MATCH (a {{id: $src}})
                        MATCH (b {{id: $dst}})
                        MERGE (a)-[r:{e.relation}]->(b)
                        SET r.evidence = $ev
                        """,
                        src=e.source,
                        dst=e.target,
                        ev=e.evidence
                    )
                # candidate relations from LLM
                for c in g.candidate_relations:
                    s.run(
                        """
                        MATCH (a {id: $src})
                        MATCH (b {id: $dst})
                        MERGE (a)-[r:CANDIDATE_REL {type: $pred}]->(b)
                        """,
                        src=c.subject,
                        dst=c.object,
                        pred=c.predicate
                    )