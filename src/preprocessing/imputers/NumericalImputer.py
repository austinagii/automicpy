class NumericalImputer:
    
    def __init__(self, strategy='mean', add_indicator=False, copy=True):
        self.strategy = strategy
        self.add_indicator = add_indicator
        self.copy=copy
        if strategy == 'trimmed_mean':
            self.imputer = TrimmedMeanImputer(add_indicator=add_indicator, copy=copy)
        elif strategy == 'mean' or strategy == 'median':
            self.imputer = CustomImputer(strategy=strategy, add_indicator=add_indicator, copy=copy)

    def impute(self, X, skip_cols=[]):
        skip_cols = [col for col in skip_cols if col in X.columns]
        cols_to_impute = [col for col in X.columns if not col in skip_cols]
        
        if not self.imputer.has_been_fitted():
            self.imputer.fit(X[cols_to_impute])
        
        X_processed = self.imputer.transform(X[cols_to_impute])
        X_skipped = X[skip_cols]
        
        processed_cols = self.imputer.get_feature_names()
        if self.add_indicator:
            processed_cols = processed_cols + [f"{col}_na" for col in processed_cols]
        
        all_cols = processed_cols + skip_cols
        all_data = np.hstack((X_processed, X_skipped))
        return pd.DataFrame(columns=all_cols, data=all_data, index=X.index)
