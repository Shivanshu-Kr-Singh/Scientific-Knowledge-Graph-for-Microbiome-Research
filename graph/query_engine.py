class QueryEngine:
    def __init__(self, loader):
        self.loader = loader

    def papers_for_taxon(self, taxon):
        q = """
        MATCH (p:Paper)-[:HAS_TAXON]->(t:Taxon)
        WHERE t.id = $taxon
        RETURN p
        """
        with self.loader.driver.session() as s:
            return list(s.run(q, taxon=taxon))

    def papers_for_disease(self, disease):
        q = """
        MATCH (p:Paper)-[:STUDIES_DISEASE]->(d:Disease)
        WHERE d.id = $disease
        RETURN p
        """
        with self.loader.driver.session() as s:
            return list(s.run(q, disease=disease))