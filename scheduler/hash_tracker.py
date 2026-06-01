import json
from pathlib import Path

PATH = Path("data/hash")
PATH.mkdir(exist_ok=True)

FILE = PATH / "paper_hashes.json"

class HashTracker:
    def load(self):
        if not FILE.exists():
            return {}
        return json.load(open(FILE))

    def save(self, data):
        json.dump(data, open(FILE, "w"), indent=2)