#Data Cleaning

import dask.dataframe as dd
import pandas as pd
data1 = pd.read_csv("trips_by_distance.csv")
data2= pd.read_csv("trips_full_data.csv")

# Fill null values according to data types
for column, dtype in new_data1_types.items(): #iterate over columns and their datatypes
    if dtype == 'Int64':
        data1[column] = data1[column].fillna(0)  #if the data type is integer Fill with 0
    elif dtype == 'object':
        data1[column] = data1[column].fillna('NULL')  #if the data type is object Fill with 'NULL'

# Fill null values according to data types
for column, dtype in new_data2_types.items(): #iterate over columns and their datatypes
    if dtype == 'Int64':
        data2[column] = data2[column].fillna(0)  #if the data type is integer Fill with 0
    elif dtype == 'object':
        data2[column] = data2[column].fillna('NULL')  #if the data type is object Fill with 'NULL'