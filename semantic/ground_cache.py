import json
from pathlib import Path

FILE = Path(__file__).parent / "cache" / "ground_cache.json"

class GroundCache:
    def load(self):
        if not FILE.exists():
            return {}
        return json.load(open(FILE))

    def save(self, db):
        json.dump(db, open(FILE, "w"), indent=2)