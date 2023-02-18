class EntityDatabase:
    def __init__(self, id):
        self.id = id
        self.entities_by_name = {}
        
    def __iter__(self):
        return iter(self.entities_by_name.items())
        
    def add(self, entity):
        if self.entities_by_name.get(entity.name) is None:
            self.entities_by_name[entity.name] = entity
            
    def get(self, entity_name):
        return self.entities_by_name.get(entity_name)
    
    def get_target(self):
        for entity in self.entities_by_name.values():
            if entity.is_target:
                return entity
    
    def get_target_variable(self):
        return self.target_variable
    
    def has_entities(self):
        return len(self.entities_by_name.values()) > 0
    
    def to_entity_set(self):
        es = ft.EntitySet(id=self.id)
        
        entity_data_by_name = {}
        for name, db_entity in self:
            entity_data_by_name[name] = db_entity.data.reset_index()    

        entity_ix_dtype_by_name = {}
        for name, db_entity in self:
            entity_ix_dtype_by_name[name] = entity_data_by_name[name][db_entity.index].dtype
        
        for name, parent_entity in self:
            for child_entity, _ in parent_entity.children:
                entity_data_by_name[child_entity.name][parent_entity.index] = entity_data_by_name[child_entity.name][parent_entity.index].astype(entity_ix_dtype_by_name[parent_entity.name])
            es.entity_from_dataframe(entity_id=parent_entity.name, dataframe=entity_data_by_name[parent_entity.name], index=parent_entity.index)
        
        # all entities have to be added to the entity set before relationships can be declared
        for _, parent_entity in self:
            for child_entity, foreign_index in parent_entity.children:
                relationship = ft.Relationship(es[parent_entity.name][parent_entity.index], es[child_entity.name][foreign_index])
                es.add_relationship(relationship)
                
        return es
