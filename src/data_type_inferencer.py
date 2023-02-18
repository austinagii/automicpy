
    
    class DataTypeInferencer:
        def __init__(self):
            self.log = LoggerFactory.get_logger(type(self).__name__)
            self.dtypes = {}

        def infer_dtypes(self, file_path, chunking_threshold=100, max_chunks=np.inf):
            dtypes = {}
            chunking_threshold = chunking_threshold*MB_SIZE
            if path.getsize(file_path) > chunking_threshold:
                self.log.info(f"File '{file_path}' exceeds chunking threshold. Performing data type inference in chunks. Max chunks: [{max_chunks}]")
                # try to estimate the number of rows per chunk
                avg_row_size = pd.read_csv(file_path, nrows=50).memory_usage(deep=True).sum() / 50
                rows_per_chunk = chunking_threshold // avg_row_size          
                for chunk_ix, chunk in enumerate(pd.read_csv(file_path, chunksize=rows_per_chunk)):
                    self.log.debug(f"Processing chunk {chunk_ix}...")
                    if chunk_ix + 1 > max_chunks:
                        break
                    chunk_dtypes = dict(((label, self.infer_dtype(values)) for (label, values) in chunk.iteritems()))
                    self.merge_and_update_dtypes(dtypes, chunk_dtypes)
            else:
                data = pd.read_csv(file_path)
                dtypes = dict(((label, self.infer_dtype(values)) for (label, values) in data.iteritems()))
            return dtypes 

        def infer_dtype(self, series):
    #         self.log.debug(series.name)
            dtype = series.dtype        
            if types.is_numeric_dtype(series.dtype):  
                has_negative_values = (series < 0).sum() > 0
                has_decimal_values = (series % 1).sum() > 0
                if types.is_integer_dtype(series.dtype):
                    # pandas never loads integers as unsigned, so we check to see 
                    # if they are any -ve values, if so then the type is indeed a
                    # signed int and we can get the smallest int that can be used
                    # to represent the series. If there are no -ve values then we 
                    # do the same but for the uint dtype
                    if has_negative_values: # has -ve values
                        dtype = self.downcast_numeric_dtype(series, 'signed')
                    else:
                        dtype = self.downcast_numeric_dtype(series, 'unsigned') 
                elif types.is_float_dtype(series.dtype):
                    if has_decimal_values:
                        dtype = self.downcast_numeric_dtype(series, 'float')
                    else:
                        if has_negative_values:
    #                         self.log.info(f"[{series.name}] has no decimal values... downcasting signed")
                            dtype = self.downcast_numeric_dtype(series, 'signed')
                        else:
    #                         self.log.info(f"[{series.name}] has no decimal values... downcasting unsigned")
                            dtype = self.downcast_numeric_dtype(series, 'unsigned')
            elif types.is_object_dtype(series.dtype):
                # object dtype seems to be a catch all for text-like fields, so 
                # categories, datetime strings, free text etc...
                dtype = np.object
                try:
                    first_valid_value = series[series[series.notna()].index[0]]
                    if self.is_categorical(series):
                        categories = series.unique()
                        categories = categories[pd.notna(categories)]
                        dtype = types.CategoricalDtype(categories=categories) 
    #                     self.log.info(f"{series.name} is generic category")
                    elif self.is_datetime_string(first_valid_value):
                        dtype = np.datetime64
    #                     self.log.info(f"{series.name} is generic datetime")
                    else:
    #                     self.log.info(f"{series.name} is generic object")
                        pass
                except KeyError:
                    self.log.error('No valid (not n/a) values in this batch.')
            else:
                self.log.info(f"{series.name} is unrecognizable type {series.dtype}")
                self.log.info(f"{series.index}")

            return dtype

        def downcast_numeric_dtype(self, series, target):
            return pd.to_numeric(series, downcast=target).dtype

        def is_datetime_string(self, string):
            is_datetime_string = False
            try:
                dateutil.parser.parse(string)
                is_datetime_string = True
            except ValueError:
                pass
            return is_datetime_string

        def is_categorical(self, series):
            info = series.describe()
            uniqueness_ratio = info['unique'] / info['count']
            return uniqueness_ratio < 0.9

        def merge_and_update_dtypes(self, a, b):
            if a is None:
                return b
            else:
                for label in b.keys():
    #                 self.log.info(f"Unioning dtypes {a.get(label)} & {b.get(label)} of column: {label}")
                    a[label] = self.merge_dtypes(a.get(label), b.get(label))
                return a

        def merge_dtypes(self, a, b):
            if a is None:
                return b

            union_dtype = None
            if types.is_numeric_dtype(a) and types.is_numeric_dtype(b):
                union_dtype = self.merge_numeric_dtypes(a, b)
            elif types.is_categorical_dtype(a) and types.is_categorical_dtype(b):
                union_dtype = self.merge_categorical_dtypes(a, b)
            elif types.is_object_dtype(a) or types.is_object_dtype(b):
                union_dtype = np.object
            elif a.dtype == b.dtype:
                union_dtype = a.dtype
            else:
                raise ValueError("Cannot union these datatypes")

            return union_dtype   

        def merge_categorical_dtypes(self, a, b):
            categories = set()
            categories.update(a.categories)
            categories.update(b.categories)
            return types.CategoricalDtype(categories, ordered=False)

        def merge_numeric_dtypes(self, a, b):
            dtype = None
            # either dtypes are of the same type (signed, unsigned, float) or they are a 
            # mix of types
            if (types.is_signed_integer_dtype(a) and types.is_signed_integer_dtype(b) or 
                 types.is_unsigned_integer_dtype(a) and types.is_unsigned_integer_dtype(b) or 
                  types.is_float_dtype(a) and types.is_float_dtype(b)):
                dtype = a if get_bits(a) > get_bits(b) else b
            else:
                if types.is_unsigned_integer_dtype(a) and types.is_signed_integer_dtype(b):
                    if get_bits(a) >= get_bits(b):
                        # if the unsigned integer requires more bits than the signed integer 
                        # then to avoid overflow errors when we convert it to a signed integer
                        # (for values that are greater than [2^(num_bits(a) - 1) - 1]) then 
                        # we need to increase the size of the signed when we convert 
                        dtype = INT_DTYPE_BY_NUM_BITS[min(get_bits(a) * 2, 64)] 
                    else:
                        dtype = b
                elif types.is_signed_integer_dtype(a) and types.is_unsigned_integer_dtype(b):
                    if get_bits(b) >= get_bits(a):
                        # same as above
                        dtype = INT_DTYPE_BY_NUM_BITS[min(get_bits(b) * 2, 64)] 
                    else:
                        dtype = a
                else: # at least one dtype is float
                    a_bits = get_bits(a)
                    b_bits = get_bits(b)

                    if a_bits == b_bits:
                        dtype = FLOAT_DTYPE_BY_NUM_BITS[min(a_bits * 2, 64)]
                    else:
                        dtype = FLOAT_DTYPE_BY_NUM_BITS[max(a_bits, b_bits)]
            return dtype

        