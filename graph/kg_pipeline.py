import json
from pathlib import Path
from graph.enhanced_graph_builder import EnhancedGraphBuilder
from nlp.enriched_record import (EnrichedPaperRecord)

class KGPipeline:
    def __init__(self):
        self.builder = EnhancedGraphBuilder()

    def run(self, enriched):
        out = []
        
        records = [EnrichedPaperRecord(**x)
            if isinstance(x,dict)
            else x
            for x in enriched]

        # Process all papers and get edges
        edges = self.builder.process_papers(records)
        
        # Create reified claims from aggregated evidence
        claims = self.builder.create_reified_claims()
        
        # Get statistics
        stats = self.builder.get_statistics()
        
        # Prepare output with edges, claims, and statistics
        out = {
            "edges": [edge.to_dict() for edge in edges],
            "claims": [claim.model_dump() for claim in claims],
            "statistics": stats
        }

        ts = __import__("datetime").datetime.now().strftime("%Y%m%d_%H%M%S")
        path = Path("data/processed") / f"graph_{ts}.json"

        with open(path, "w", encoding="utf-8") as fp:
            json.dump(out, fp, indent=2, default=str)

        return path