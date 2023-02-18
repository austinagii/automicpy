class CategoryImputer:
    @staticmethod
    def impute(X, skip_cols=None):
        if isinstance(X, pd.Series):
            if not X.name in skip_cols:
                CategoryImputer.fill_mode(X)
        elif isinstance(X, pd.DataFrame):
            for label, data in X.iteritems():
                if not label in skip_cols:
                    CategoryImputer.fill_mode(data)
        return X
    
    @staticmethod 
    def fill_mode(X):
        X.fillna(X[X.notna()].mode()[0], inplace=True)
        
