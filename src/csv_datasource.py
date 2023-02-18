class CSVDataSourceConfig:
    
    class EntityConfig:
        def __init__(self, name, file, index=None, is_target=False, target_col=None):
            self.name = name
            self.file = file
            self.index = index
            self.is_target = is_target if not is_target is None else False
            self.target_col = target_col
            self.children = []
            
        def add_child(self, entity, key=None):
            # assume the name of the key in the child entity is the same 
            # as the name of the index of the parent if none is provided
            if key is None:
                key = self.index
                
            self.children.append((entity, key))
            
        def has_index(self):
            return (not self.index is None)
            
            
    def __init__(self, id, data_dir):
        self.id = id
        self.data_dir = data_dir
        self.entity_config_by_name = {}
        
    def __iter__(self):
        return iter(self.entity_config_by_name.values())
        
    def add_entity(self, entity):
        self.entity_config_by_name[entity.name] = entity
        
    def get_target_entity(self):
        for entity_config in self.entities.values():
            if entity_config.is_target == True:
                return entity_config
            
    @staticmethod
    def from_string(config_string):
        config = json.loads(config_string)
        
        data_config =  CSVDataSourceConfig(config["id"], config["data_directory"])
        for entity_config in config["entities"]:
            csv_entity_config = CSVDataSourceConfig.EntityConfig(entity_config["name"], 
                                                                 entity_config["file"], 
                                                                 index=entity_config.get("index"), 
                                                                 is_target=entity_config.get("is_target"), 
                                                                 target_col=entity_config.get("target_col"))
            if not entity_config.get("children") is None:    
                for child_config in entity_config["children"]:
                    csv_entity_config.add_child(*child_config)    
            data_config.add_entity(csv_entity_config)
        return data_config