# Pandas

## Installation
```bash
pip install pandas
```

## Import
```python
import pandas as pd
import numpy as np
```

## Data Structures

### Series (1D)
```python
# From list
s = pd.Series([1, 2, 3, 4])

# With custom index
s = pd.Series([1, 2, 3], index=['a', 'b', 'c'])

# From dictionary
s = pd.Series({'a': 1, 'b': 2, 'c': 3})
```

### DataFrame (2D)
```python
# From dictionary
df = pd.DataFrame({
    'name': ['Alice', 'Bob', 'Charlie'],
    'age': [25, 30, 35],
    'city': ['NYC', 'LA', 'Chicago']
})

# From list of dictionaries
df = pd.DataFrame([
    {'name': 'Alice', 'age': 25},
    {'name': 'Bob', 'age': 30}
])

# From NumPy array
df = pd.DataFrame(np.random.randn(4, 3), columns=['A', 'B', 'C'])
```

## Data Input/Output

### Reading Data
```python
# CSV files
df = pd.read_csv('file.csv')
df = pd.read_csv('file.csv', index_col=0)  # use first column as index
df = pd.read_csv('file.csv', sep=';')      # custom separator

# Excel files
df = pd.read_excel('file.xlsx')
df = pd.read_excel('file.xlsx', sheet_name='Sheet1')

# JSON
df = pd.read_json('file.json')

# SQL
df = pd.read_sql('SELECT * FROM table', connection)

# Parquet
df = pd.read_parquet('file.parquet')
```

### Writing Data
```python
df.to_csv('output.csv', index=False)
df.to_excel('output.xlsx', index=False)
df.to_json('output.json')
df.to_parquet('output.parquet')
```

## DataFrame Inspection

### Basic Info
```python
df.head()              # first 5 rows
df.tail(3)             # last 3 rows
df.info()              # data types and memory usage
df.describe()          # summary statistics
df.shape               # (rows, columns)
df.columns             # column names
df.index               # row indices
df.dtypes              # data types
```

### Quick Stats
```python
df.nunique()           # unique values per column
df.isnull().sum()      # count of missing values
df.value_counts()      # frequency counts (Series)
df['col'].value_counts()  # frequency counts for column
```

## Data Selection

### Column Selection
```python
df['name']             # single column (Series)
df[['name', 'age']]    # multiple columns (DataFrame)
df.name                # single column (attribute access)
```

### Row Selection
```python
df.iloc[0]             # first row by position
df.iloc[0:3]           # first 3 rows
df.loc['row_name']     # row by label
df.loc[0:2, 'name':'age']  # rows and columns by label
```

### Boolean Indexing
```python
df[df['age'] > 30]                    # rows where age > 30
df[df['name'].isin(['Alice', 'Bob'])] # rows where name in list
df[(df['age'] > 25) & (df['age'] < 35)]  # multiple conditions
df.query('age > 30 and city == "NYC"')  # query string
```

## Data Manipulation

### Adding/Modifying Columns
```python
df['new_col'] = 0                    # add column with constant
df['age_squared'] = df['age'] ** 2   # derived column
df['full_name'] = df['first'] + ' ' + df['last']  # string concatenation
df.assign(new_col=df['age'] * 2)     # add column (returns new DataFrame)
```

### Dropping Data
```python
df.drop('column_name', axis=1)       # drop column
df.drop(['col1', 'col2'], axis=1)    # drop multiple columns
df.drop(0, axis=0)                   # drop row by index
df.dropna()                          # drop rows with NaN
df.dropna(axis=1)                    # drop columns with NaN
df.drop_duplicates()                 # remove duplicate rows
```

### Handling Missing Data
```python
df.isnull()                          # boolean mask of NaN values
df.notnull()                         # boolean mask of non-NaN values
df.fillna(0)                         # fill NaN with 0
df.fillna(method='ffill')            # forward fill
df.fillna(method='bfill')            # backward fill
df.fillna(df.mean())                 # fill with column means
df.interpolate()                     # interpolate missing values
```

## Data Aggregation and Grouping

### Basic Aggregation
```python
df.sum()               # sum of each column
df.mean()              # mean of each column
df.median()            # median
df.std()               # standard deviation
df.min(), df.max()     # min/max
df.count()             # non-null count
```

### GroupBy Operations
```python
# Group by single column
df.groupby('category').mean()
df.groupby('category')['value'].sum()

# Group by multiple columns
df.groupby(['cat1', 'cat2']).agg({
    'value1': 'sum',
    'value2': 'mean'
})

# Apply custom functions
df.groupby('category').apply(lambda x: x.max() - x.min())

# Multiple aggregations
df.groupby('category').agg(['sum', 'mean', 'count'])
```

## Data Transformation

### Apply Functions
```python
df['col'].apply(lambda x: x * 2)     # apply to Series
df.apply(lambda x: x.max() - x.min(), axis=1)  # apply to rows
df.apply(np.sum, axis=0)             # apply to columns
```

### String Operations
```python
df['name'].str.upper()               # uppercase
df['name'].str.lower()               # lowercase
df['name'].str.len()                 # length
df['name'].str.contains('pattern')   # contains pattern
df['name'].str.replace('old', 'new') # replace
df['name'].str.split(',')            # split string
df['name'].str.extract('(\d+)')      # extract with regex
```

### Datetime Operations
```python
df['date'] = pd.to_datetime(df['date'])  # convert to datetime
df['date'].dt.year                       # extract year
df['date'].dt.month                      # extract month
df['date'].dt.dayofweek                  # day of week (0=Monday)
df['date'].dt.strftime('%Y-%m')          # format as string
```

## Merging and Joining

### Concatenation
```python
pd.concat([df1, df2])                # vertical concatenation
pd.concat([df1, df2], axis=1)        # horizontal concatenation
pd.concat([df1, df2], ignore_index=True)  # reset index
```

### Merging
```python
# Inner join (default)
pd.merge(df1, df2, on='key')

# Left join
pd.merge(df1, df2, on='key', how='left')

# Outer join
pd.merge(df1, df2, on='key', how='outer')

# Multiple keys
pd.merge(df1, df2, on=['key1', 'key2'])

# Different column names
pd.merge(df1, df2, left_on='key1', right_on='key2')
```

## Reshaping Data

### Pivot Tables
```python
# Basic pivot
df.pivot(index='date', columns='category', values='value')

# Pivot table with aggregation
df.pivot_table(
    values='value',
    index='date',
    columns='category',
    aggfunc='sum'
)
```

### Melt (Wide to Long)
```python
df.melt(
    id_vars=['id', 'name'],          # columns to keep
    value_vars=['col1', 'col2'],     # columns to melt
    var_name='variable',             # name for variable column
    value_name='value'               # name for value column
)
```

### Stack/Unstack
```python
df.stack()                           # pivot columns to rows
df.unstack()                         # pivot rows to columns
df.unstack(level=0)                  # unstack specific level
```

## Sorting

```python
df.sort_values('column')             # sort by single column
df.sort_values(['col1', 'col2'])     # sort by multiple columns
df.sort_values('col', ascending=False)  # descending order
df.sort_index()                      # sort by index
df.nlargest(5, 'column')            # top 5 values
df.nsmallest(5, 'column')           # bottom 5 values
```

## Advanced Operations

### Rolling Windows
```python
df['value'].rolling(window=3).mean()     # 3-period moving average
df['value'].rolling(window=7).sum()      # 7-period rolling sum
df['value'].expanding().mean()           # expanding window mean
```

### Resampling (Time Series)
```python
# Resample daily to monthly
df.resample('M').mean()
df.resample('Q').sum()                   # quarterly
df.resample('W').last()                  # weekly, last value
```

### MultiIndex
```python
# Create MultiIndex
df.set_index(['col1', 'col2'])

# Access MultiIndex data
df.loc[('level1', 'level2')]
df.xs('level1', level=0)

# Reset MultiIndex
df.reset_index()
```

### Performance Optimization
```python
# Use categorical data for repeated strings
df['category'] = df['category'].astype('category')

# Use appropriate data types
df['int_col'] = df['int_col'].astype('int32')  # smaller int type
df['float_col'] = df['float_col'].astype('float32')

# Use chaining for multiple operations
(df
 .dropna()
 .groupby('category')
 .sum()
 .sort_values('value', ascending=False)
)
```

## Common Patterns

### Conditional Logic
```python
# np.where for simple conditions
df['new_col'] = np.where(df['age'] > 30, 'Senior', 'Junior')

# np.select for multiple conditions
conditions = [df['age'] < 25, df['age'] < 40, df['age'] >= 40]
choices = ['Young', 'Middle', 'Senior']
df['age_group'] = np.select(conditions, choices)
```

### Window Functions
```python
df['rank'] = df['value'].rank()
df['pct_rank'] = df['value'].rank(pct=True)
df['cumsum'] = df['value'].cumsum()
df['lag'] = df['value'].shift(1)        # previous value
df['lead'] = df['value'].shift(-1)      # next value
```

### Memory Usage
```python
df.memory_usage(deep=True)              # memory usage by column
df.info(memory_usage='deep')            # detailed memory info
```