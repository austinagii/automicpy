class CategoryEncoder:
    # TODO :: Make sure these categories are binary cateogries in both pandas and featuretools 
    #         For better inferences later
    @staticmethod
    def encode(data, strategy="select_best"):
        
        categories = list(data.dtype.categories)
        n_categories = len(categories)
        df = pd.DataFrame({data.name: data.values}, index=data.index)
        if strategy == "select_best":
            if n_categories < 8:
                data_tr = ce.OneHotEncoder(cols=[data.name], use_cat_names=True).fit_transform(df)
            elif n_categories < 64:
                data_tr = ce.HashingEncoder(cols=[data.name], n_components=CategoryEncoder.calculate_n_feature_bins(n_categories), max_process=4).fit_transform(df)            
                data_tr.index = data.index
                data_tr.columns = [f"{data.name}_{ix + 1}" for ix, _ in enumerate(data_tr.columns)]
                # TODO:: Column names come out at 'colx', where x in range(1, n_buckets), change this to '{column_name}_bucket_x'
            else:
                data_tr = ce.BinaryEncoder(cols=[data.name]).fit_transform(df)
        elif strategy == "one_hot" :
            data_tr = ce.OneHotEncoder(cols=[data.name], use_cat_names=True).fit_transform(df)
            
        return data_tr
    
    @staticmethod
    def calculate_n_feature_bins(n_categories):
        return 8
