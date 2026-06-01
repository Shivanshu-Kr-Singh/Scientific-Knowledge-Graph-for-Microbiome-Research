from semantic.entity_registry import EntityRegistry

class SchemaInducer:
    LIMIT = 100

    def __init__(self):
        self.reg = EntityRegistry()

    def observe(self, entity):
        self.reg.update(entity.entity_type)
        db = self.reg.load()

        return db.get(entity.entity_type, 0) >= self.LIMIT