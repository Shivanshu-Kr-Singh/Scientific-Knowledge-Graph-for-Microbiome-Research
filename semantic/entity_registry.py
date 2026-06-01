import json
from pathlib import Path

FILE = Path(__file__).parent / "cache" / "entity_registry.json"

class EntityRegistry:
    def load(self):
        if not FILE.exists():
            return {}
        return json.load(open(FILE))

    def update(self, entity_type):
        db = self.load()
        db.setdefault(entity_type, 0)
        db[entity_type] += 1

        json.dump(db, open(FILE, "w"), indent=2)