class CustomImputer(BaseEstimator, TransformerMixin):
    def __init__(self, strategy='mean', add_indicator=False, copy=False):
        self.indicator = MissingIndicator(features='all')
        self.strategy = strategy
        self.add_indicator = add_indicator
        self.copy = copy
        self.imputer = SimpleImputer(strategy=strategy, copy=copy)
        self.is_fitted = False
    
    def fit(self, X, y=None):
        self.imputer = self.imputer.fit(X, y)
        self.feature_names = list(X.columns)
        self.is_fitted = True
        return self
    
    def transform(self, X, y=None):
        X_tr = self.imputer.transform(X)
        if not self.add_indicator:
            return X_tr
        else:
            X_missing = self.indicator.fit_transform(X)
            return np.hstack((X_tr, X_missing))
    
    def get_feature_names(self):
        return self.feature_names
    
    def has_been_fitted(self):
        return self.is_fitted
        
    
