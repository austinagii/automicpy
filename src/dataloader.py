
            
class DataLoader:
    
    def __init__(self, infer_dtypes=True, chunking_threshold=100, max_chunks=-1):
        self.log = LoggerFactory.get_logger(type(self).__name__)        
        self.infer_dtypes = infer_dtypes
        self.chunking_threshold = chunking_threshold
        self.max_chunks = np.inf if max_chunks == -1 else max_chunks 

    def load(self, config):
        database = EntityDatabase(config.id)
        
        for entity_config in config:
            file_path = path.join(config.data_dir, entity_config.file)
            self.log.info(f"Creating entity '{entity_config.name}' from file '{file_path}'")
            
            if self.infer_dtypes:
                self.log.info("Performing data type inference...")
                dtypes = self.DataTypeInferencer().infer_dtypes(file_path, chunking_threshold=self.chunking_threshold, max_chunks=self.max_chunks)
                date_cols=[]
                for label, dtype in dtypes.items():
                    if types.is_datetime64_any_dtype(dtype):
                        dates_cols.append(label)
                for date_col in date_cols:
                    del types[date_col]
                self.log.info("Data type inference completed. Loading entity data into memory...")
                start = timer()
                data = pd.read_csv(file_path, dtype=dtypes, parse_dates=date_cols, infer_datetime_format=True)
                end = timer()
                self.log.info(f"Data loading completed in {end - start: .2f} secs")
            else:
                self.log.info("Loading entity data into memory...")
                start = timer()
                data = pd.read_csv(file_path)
                end = timer()
                self.log.info(f"Data loading completed in {end - start: .2f} secs")
                
            if entity_config.has_index():
                self.log.debug(f"Setting entity index to {entity_config.index}")
                data.set_index(entity_config.index, inplace=True)
            else:
                self.log.info(f"No index specified for entity '{entity.name}', assigning default index...")
                data.reset_index(inplace=True)
                index = f"{entity_config.name}_id"
                data.rename(columns={'Index': index}, inplace=True)
                entity_config.index = index
                
            database.add(Entity(name=entity_config.name, 
                                data=data, 
                                index=entity_config.index,
                                is_target=entity_config.is_target,
                                target_col=entity_config.target_col))
            self.log.info("-----------------------------------------------------------------------------------------------")
        self.log.info("Defining entity relationships...")
        for entity_config in config: 
            for child_name, key in entity_config.children:
                self.log.info(f"Adding child '{child_name}' for entity '{entity_config.name}'")
                database.get(entity_config.name).add_child(database.get(child_name), key)
        self.log.info("-----------------------------------------------------------------------------------------------")
        database.target_variable = database.get_target().data[database.get_target().target_col]
        database.get_target().data = database.get_target().data.drop(database.get_target().target_col, axis=1)
        return database
