import yaml
from pathlib import Path
from semantic.ground_cache import GroundCache
from semantic.llm_grounder import LLMGrounder

REG = Path(__file__).parent / "ontology" / "registry.yaml"

class OntologyGrounder:
    def __init__(self):
        self.reg = yaml.safe_load(open(REG))
        self.cache = GroundCache()
        self.llm = LLMGrounder()
        self.alias_map = {}

        for canon, cfg in self.reg.items():
            self.alias_map[canon.lower()] = canon
            
            for a in cfg.get("aliases", []):
                self.alias_map[a.lower()] = canon


    def ground(self, entity, llm=None):
        db = self.cache.load()
        key = entity.name

        if key in db:
            return db[key]

        et = (entity.entity_type or "").lower()
        et = self.alias_map.get(et, et)

        if et in self.reg:
            out = {
                "canonical": entity.name,
                "ontology": self.reg[et]["ontology"]
            }
            db[key] = out
            self.cache.save(db)
            return out

        if llm:
            out = self.llm.resolve(entity)
            db[key] = out
            self.cache.save(db)
            return out

        return None