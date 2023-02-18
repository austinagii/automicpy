class NumericalDataPreprocessor:
    def __init__(self,
                 drop_na=False,
                 drop_na_threshold=0.3,
                 add_indicator=True,
                 imputation_strategy='trimmed_mean',
                 scale_data=True,
                 scaling_strategy='standard'):
        
        self.imputer = NumericalImputer(strategy=imputation_strategy, add_indicator=add_indicator)
        self.scaler = NumericalScaler(strategy=scaling_strategy)
        self.log = LoggerFactory.get_logger(type(self).__name__)
        
        self.drop_na = drop_na
        self.drop_na_threshold=0.3
        self.add_indicator = add_indicator
        self.imputation_strategy = imputation_strategy
        self.scale_data = scale_data
        self.scaling_strategy = scaling_strategy
    
    def process(self, X, skip_cols=[]):
        # impute missing values
        self.log.info(f"Performing '{self.imputation_strategy}' imputation on the input data...")
        X_tr = self.imputer.impute(X, skip_cols=skip_cols)
        
        if self.scale_data:
            na_indicator_variables = [col for col in X_tr.columns if '_na' in col]
            if len(na_indicator_variables) > 0:
                skip_cols = skip_cols + na_indicator_variables
                self.log.info(f"The following missing value indicator features have been excluded from preprocessing: {na_indicator_variables}")
                
            self.log.info(f"Performing '{self.scaling_strategy}' scaling on the input data...")
            X_tr = self.scaler.scale(X_tr, skip_cols=skip_cols)

        self.log.info("-----------------------------------------------------------------------------------------------")
        # convert to dataframe if series
        if isinstance(X_tr, pd.Series):
            X_tr = pd.DataFrame({X_tr.name: X_tr}, index=X.index)
        return X_tr
