_registry_ = {}

def add_entry(key, fn):
    global _registry_
    if key in _registry_:
        raise ValueError("Key {} already in registery".format(key))
    else:
        _registry_[key] = fn

def get_entry(key):
    global _registry_
    if key in _registry_:
        return _registry_[key]

    return None
                        
