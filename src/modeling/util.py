MODEL_REGISTRY = {}


def register_model(name):
    def add_to_registry(cls):
        MODEL_REGISTRY[name] = cls
        return cls
    return add_to_registry


TOKEN_POOLER_REGISTRY = {}


def register_token_pooler(name):
    def add_to_registry(cls):
        TOKEN_POOLER_REGISTRY[name] = cls
        return cls
    return add_to_registry
