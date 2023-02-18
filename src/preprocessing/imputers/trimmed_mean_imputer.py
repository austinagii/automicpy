class TrimmedMeanImputer(BaseEstimator, TransformerMixin):
    def __init__(self, trim_portion=0.1, add_indicator=False, copy=False):
        self.means_by_column_name = {}
        self.trim_portion = trim_portion
        self.copy = copy
        self.add_indicator = add_indicator
        self.is_fitted = False
        
    def fit(self, X, y=None):
        for label, values in X.iteritems():
            self.means_by_column_name[label] = stats.trim_mean(values[values.notna()], self.trim_portion)
        self.feature_names = list(X.columns)
        self.is_fitted = True
        return self
    
    def transform(self, X, y=None):
        if self.copy:
            X = X.copy(deep=True)
        
        if not self.add_indicator:
            for label, values in X.iteritems():
                fill_value = self.means_by_column_name.get(label)
                if not fill_value is None:
                    values.fillna(fill_value, inplace=True)
            return X
        else:
            na_indicators = []
            for label, values in X.iteritems():
                fill_value = self.means_by_column_name.get(label)
                if not fill_value is None:
                    na_indicators.append(values.isna().astype('int8'))
                    values.fillna(fill_value, inplace=True)

            if self.add_indicator:
                na_indicator_dataframe = pd.concat(na_indicators, axis=1)
    #         na_indicator_dataframe.columns = [f"{name}_na" for name in X.columns if name in self.means_by_column_name.keys()]
            return pd.concat([X, na_indicator_dataframe], axis=1).values
    
    def get_fitted_means(self):
        return self.means_by_column_name
    
    def get_feature_names(self):
        return self.feature_names

    def has_been_fitted(self):
        return self.is_fitted
    