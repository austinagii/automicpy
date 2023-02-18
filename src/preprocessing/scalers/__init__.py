class NumericalScaler:
    
    scaler_by_strategy = {
        'standard': StandardScaler,
        'min_max': MinMaxScaler
    }
    
    def __init__(self, strategy='standard', copy=True):
        self.scaler = self.scaler_by_strategy[strategy]()
        self.copy = copy
    
    def scale(self, X, skip_cols=[]):
        skip_cols = [col for col in skip_cols if col in X.columns]
        if self.copy:
            X = X.copy(deep=True)
        
        cols_to_scale = [col for col in X.columns if not col in skip_cols]
        if not self.has_been_fitted():
            self.scaler.fit(X[cols_to_scale])
        
        X_scaled = self.scaler.transform(X[cols_to_scale])
        X_skipped = X[skip_cols]
        
        all_cols = cols_to_scale + skip_cols
        all_data = np.hstack((X_scaled, X_skipped.values))
        return pd.DataFrame(columns=all_cols, data=all_data, index=X.index)
    
    def has_been_fitted(self):
        fitted = True
        try:
            validation.check_is_fitted(self.scaler, attributes=['scale_'])
        except exceptions.NotFittedError:
            fitted = False
        return fitted
    