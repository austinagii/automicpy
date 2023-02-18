class Entity:            
    def __init__(self, name, data, index, is_target=False, target_col=None, children=None):
        self.name = name
        self.index = index
        self.data = data
        self.is_target = is_target
        self.target_col = target_col
        if children is None:
            self.children = []
        else:
            self.children = children 
        
    def add_child(self, entity, key):
        self.children.append((entity, key))
        