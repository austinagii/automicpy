class DataPreprocessor:
    def __init__(self, 
                 drop_na=True,
                 drop_na_threshold=0.30,
                 add_numerical_data_na_indicator=False,
                 numerical_data_imputation_strategy="trimmed_mean",
                 scaling_strategy="standard",
                 impute_categorical_features=False,
                 category_encoding_strategy="select_best"):
        self.log = LoggerFactory.get_logger(type(self).__name__)
        self.drop_na = drop_na
        self.drop_na_threshold = drop_na_threshold
        self.add_numerical_data_na_indicator = add_numerical_data_na_indicator
        self.numerical_data_imputation_strategy = numerical_data_imputation_strategy
        self.scaling_strategy = scaling_strategy
        self.impute_categorical_features=impute_categorical_features
        self.category_encoding_strategy=category_encoding_strategy
        self.numerical_preprocessor_by_entity_name = {}
        self.categorical_preprocessor_by_entity_name = {}
  
    def preprocess(self, database):
        self.log.info(f"Initiating preprocessing for dataset: '{database.id}' ...")
        
        skip_cols = set()
        for _, entity in database:
            skip_cols.add(entity.index)
            for _, parent_foreign_key in entity.children:
                skip_cols.add(parent_foreign_key)
        self.log.info(f"The following fields will be excluded from preprocessing: {skip_cols}")
        
        for _, entity in database:
            self.log.info(f"Preprocessing features for entity: '{entity.name}'")
            self.log.info("Processing numerical features...")
            numerical_preprocessor = self.numerical_preprocessor_by_entity_name.get(entity.name)
            if numerical_preprocessor is None:
                numerical_preprocessor = NumericalDataPreprocessor(drop_na=self.drop_na,
                                                                   drop_na_threshold=self.drop_na_threshold,
                                                                   add_indicator=self.add_numerical_data_na_indicator,
                                                                   imputation_strategy=self.numerical_data_imputation_strategy,
                                                                   scaling_strategy=self.scaling_strategy)
                self.numerical_preprocessor_by_entity_name[entity.name] = numerical_preprocessor
            X_numerical = numerical_preprocessor.process(entity.data.select_dtypes('number'), skip_cols=list(skip_cols))
            
            self.log.info("Processing categorical features...")
            categorical_preprocessor = self.categorical_preprocessor_by_entity_name.get(entity.name)
            if categorical_preprocessor is None:
                categorical_preprocessor = CategoricalDataPreprocessor(impute_na=self.impute_categorical_features,
                                                                       encoding_strategy=self.category_encoding_strategy)
                self.categorical_preprocessor_by_entity_name[entity.name] = categorical_preprocessor
            X_categorical = categorical_preprocessor.process(entity.data.select_dtypes('category'), skip_cols=list(skip_cols))
            entity.data = pd.concat([X_numerical, X_categorical], axis=1)
        return database