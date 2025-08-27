# Polars Cheat Sheet

Polars is a blazingly fast DataFrame library implemented in Rust, offering lazy and eager execution, multi-threading, and a powerful expression API for Python.

## Quick Start

### Installation
```bash
pip install polars
# With optional dependencies
pip install polars[all]
# For cloud storage support  
pip install polars[aws,azure,gcp]
```

### Basic Import
```python
import polars as pl
```

## DataFrame Creation

### From Dictionary
```python
df = pl.DataFrame({
    "name": ["Alice", "Bob", "Charlie"],
    "age": [25, 30, 35],
    "city": ["NY", "LA", "Chicago"]
})
```

### From Lists
```python
df = pl.DataFrame([
    ["Alice", 25, "NY"],
    ["Bob", 30, "LA"],
    ["Charlie", 35, "Chicago"]
], schema=["name", "age", "city"])
```

### From NumPy/Pandas
```python
import numpy as np
import pandas as pd

# From NumPy
df = pl.from_numpy(np_array, schema=["col1", "col2"])

# From Pandas
df = pl.from_pandas(pd_df)
```

## Data Loading & Saving

### CSV Operations
```python
# Read CSV
df = pl.read_csv("data.csv")

# Scan CSV (lazy loading)
lazy_df = pl.scan_csv("data.csv")

# Write CSV
df.write_csv("output.csv")

# Advanced reading with options
df = pl.read_csv(
    "data.csv",
    separator=",",
    has_header=True,
    dtypes={"age": pl.Int64, "salary": pl.Float64},
    null_values=["NULL", "NA", ""],
    try_parse_dates=True
)
```

### Parquet Operations
```python
# Read Parquet
df = pl.read_parquet("data.parquet")

# Scan Parquet (lazy loading)
lazy_df = pl.scan_parquet("data.parquet")

# Write Parquet
df.write_parquet("output.parquet")

# Read with filters (predicate pushdown)
df = pl.scan_parquet("data.parquet").filter(
    pl.col("age") > 25
).collect()
```

### JSON Operations
```python
# Read JSON
df = pl.read_json("data.json")

# Read NDJSON (newline-delimited JSON)
df = pl.read_ndjson("data.ndjson")

# Write JSON
df.write_json("output.json")
df.write_ndjson("output.ndjson")
```

### Cloud Storage
```python
# Read from S3
df = pl.read_parquet("s3://bucket/file.parquet")

# Read from Azure
df = pl.read_csv("azure://container/file.csv")

# Read from Google Cloud
df = pl.read_parquet("gs://bucket/file.parquet")
```

## Data Selection & Filtering

### Column Selection
```python
# Select columns
df.select(["name", "age"])
df.select(pl.col("name", "age"))

# Select with expressions
df.select([
    pl.col("name"),
    pl.col("age").alias("years"),
    pl.col("salary") * 1.1  # 10% increase
])

# Select by data type
df.select(pl.col(pl.Utf8))  # String columns
df.select(pl.col(pl.Int64))  # Integer columns
```

### Row Filtering
```python
# Basic filtering
df.filter(pl.col("age") > 25)

# Multiple conditions
df.filter(
    (pl.col("age") > 25) & 
    (pl.col("city") == "NY")
)

# String operations
df.filter(pl.col("name").str.contains("A"))
df.filter(pl.col("name").str.starts_with("A"))
df.filter(pl.col("name").str.ends_with("e"))

# Null filtering
df.filter(pl.col("age").is_not_null())
df.filter(pl.col("name").is_null())
```

### First/Last Rows
```python
df.head(5)        # First 5 rows
df.tail(5)        # Last 5 rows
df.limit(10)      # First 10 rows
df.slice(5, 10)   # Rows 5-14
```

## Data Transformation

### Adding/Modifying Columns
```python
# Add new columns
df = df.with_columns([
    pl.col("age").alias("years"),
    (pl.col("salary") * 1.1).alias("new_salary"),
    pl.lit("employee").alias("type")
])

# Conditional column creation
df = df.with_columns(
    pl.when(pl.col("age") > 30)
    .then(pl.lit("Senior"))
    .otherwise(pl.lit("Junior"))
    .alias("level")
)
```

### String Operations
```python
df = df.with_columns([
    pl.col("name").str.to_uppercase().alias("name_upper"),
    pl.col("name").str.to_lowercase().alias("name_lower"),
    pl.col("name").str.len().alias("name_length"),
    pl.col("text").str.replace("old", "new"),
    pl.col("text").str.split(" ").alias("words")
])
```

### Mathematical Operations
```python
df = df.with_columns([
    pl.col("value") + 10,
    pl.col("value") * 2,
    pl.col("value").sqrt(),
    pl.col("value").log(),
    pl.col("value").abs(),
    pl.col("value").round(2)
])
```

### Date/Time Operations
```python
df = df.with_columns([
    pl.col("date").dt.year().alias("year"),
    pl.col("date").dt.month().alias("month"),
    pl.col("date").dt.day().alias("day"),
    pl.col("date").dt.weekday().alias("weekday"),
    pl.col("timestamp").dt.hour().alias("hour")
])
```

## Aggregation & Grouping

### Basic Aggregations
```python
# Single column aggregations
df.select([
    pl.col("age").sum(),
    pl.col("age").mean(),
    pl.col("age").median(),
    pl.col("age").min(),
    pl.col("age").max(),
    pl.col("age").std(),
    pl.col("age").var()
])
```

### Group By Operations
```python
# Basic groupby
result = df.group_by("city").agg([
    pl.col("age").mean().alias("avg_age"),
    pl.col("salary").sum().alias("total_salary"),
    pl.col("name").count().alias("count")
])

# Multiple groupby columns
result = df.group_by(["city", "department"]).agg([
    pl.col("salary").mean(),
    pl.col("age").max()
])

# Complex aggregations
result = df.group_by("city").agg([
    pl.col("age").filter(pl.col("age") > 25).mean(),
    pl.col("salary").quantile(0.95),
    pl.col("name").n_unique()
])
```

### Window Functions
```python
df = df.with_columns([
    pl.col("salary").sum().over("department").alias("dept_total"),
    pl.col("salary").rank().over("department").alias("salary_rank"),
    pl.col("age").mean().over("city").alias("city_avg_age"),
    pl.col("sales").rolling_mean(window_size=3).alias("rolling_avg")
])
```

## Joining Operations

### Basic Joins
```python
# Inner join
result = df1.join(df2, on="id", how="inner")

# Left join
result = df1.join(df2, on="id", how="left")

# Outer join
result = df1.join(df2, on="id", how="outer")

# Different column names
result = df1.join(df2, left_on="user_id", right_on="id")
```

### Advanced Joins
```python
# Multiple columns
result = df1.join(df2, on=["id", "date"])

# As-of join (temporal join)
result = df1.join_asof(
    df2,
    on="timestamp",
    strategy="backward"  # or "forward", "nearest"
)

# Cross join
result = df1.join(df2, how="cross")

# Semi and anti joins
result = df1.join(df2, on="id", how="semi")  # Keep matching rows
result = df1.join(df2, on="id", how="anti")  # Keep non-matching rows
```

## Data Manipulation

### Sorting
```python
# Single column
df.sort("age")
df.sort("age", descending=True)

# Multiple columns
df.sort(["city", "age"], descending=[False, True])

# By expression
df.sort(pl.col("salary") * pl.col("bonus"))
```

### Unique Values
```python
# Get unique rows
df.unique()

# Unique based on subset
df.unique(subset=["city"])

# Drop duplicates
df = df.unique(maintain_order=True)
```

### Reshaping
```python
# Pivot
pivot_df = df.pivot(
    values="salary",
    index="name", 
    columns="year",
    aggregate="sum"
)

# Melt (unpivot)
melted = df.melt(
    id_vars=["name", "city"],
    value_vars=["salary_2022", "salary_2023"],
    variable_name="year",
    value_name="salary"
)
```

### Handling Missing Data
```python
# Fill null values
df = df.fill_null(0)  # Fill with constant
df = df.fill_null(pl.col("age").median())  # Fill with median

# Drop null values
df = df.drop_nulls()  # Drop rows with any nulls
df = df.drop_nulls(subset=["age", "salary"])  # Drop if specific columns null

# Forward/backward fill
df = df.fill_null(strategy="forward")
df = df.fill_null(strategy="backward")
```

## Lazy Evaluation

### Creating Lazy Frames
```python
# From scan operations
lazy_df = pl.scan_csv("large_file.csv")

# Convert DataFrame to LazyFrame
lazy_df = df.lazy()
```

### Lazy Operations
```python
# Chain operations without execution
query = (
    pl.scan_csv("data.csv")
    .filter(pl.col("age") > 25)
    .group_by("city")
    .agg(pl.col("salary").mean())
    .sort("salary", descending=True)
)

# Execute the query
result = query.collect()

# Show query plan
print(query.explain())
```

### Streaming
```python
# For very large datasets
result = (
    pl.scan_csv("huge_file.csv")
    .filter(pl.col("value") > 100)
    .group_by("category")
    .agg(pl.col("amount").sum())
    .collect(streaming=True)
)
```

## Column Operations

### List Operations
```python
df = df.with_columns([
    pl.col("list_col").list.len().alias("list_length"),
    pl.col("list_col").list.get(0).alias("first_item"),
    pl.col("list_col").list.sum().alias("list_sum"),
    pl.col("list_col").list.unique().alias("unique_items")
])

# Explode lists to rows
df = df.explode("list_col")
```

### Struct Operations
```python
# Create struct
df = df.with_columns(
    pl.struct(["name", "age"]).alias("person")
)

# Extract struct fields
df = df.with_columns([
    pl.col("person").struct.field("name"),
    pl.col("person").struct.field("age")
])
```

### Categorical Operations
```python
# Convert to categorical
df = df.with_columns(
    pl.col("category").cast(pl.Categorical)
)

# Get categories
df.select(pl.col("category").cat.get_categories())
```

## Performance Optimization

### Data Types
```python
# Optimize data types for memory
df = df.with_columns([
    pl.col("small_int").cast(pl.Int8),     # vs Int64
    pl.col("category").cast(pl.Categorical), # vs String
    pl.col("flag").cast(pl.Boolean)        # vs String
])
```

### Query Optimization
```python
# Use lazy evaluation for large datasets
lazy_result = (
    pl.scan_parquet("large_data.parquet")
    .select(["needed_col1", "needed_col2"])  # Project early
    .filter(pl.col("date") > "2023-01-01")   # Filter early
    .collect()
)

# Use predicate pushdown with file formats
df = pl.scan_parquet("data.parquet").filter(
    pl.col("partition_col") == "value"
).collect()
```

### Memory Management
```python
# For memory-constrained environments
df = df.rechunk()  # Optimize memory layout

# Process in chunks
for batch in pl.read_csv_batched("large_file.csv", batch_size=10000):
    result = process_batch(batch)
```

## Integration with Other Libraries

### Pandas Integration
```python
# Convert to pandas
pandas_df = df.to_pandas()

# From pandas with zero-copy (when possible)
polars_df = pl.from_pandas(pandas_df, rechunk=False)
```

### NumPy Integration
```python
# To NumPy
numpy_array = df.to_numpy()

# From NumPy
df = pl.from_numpy(numpy_array, schema=["col1", "col2"])
```

### Arrow Integration
```python
# To Arrow
arrow_table = df.to_arrow()

# From Arrow
df = pl.from_arrow(arrow_table)
```

## SQL Interface

### Basic SQL
```python
# Execute SQL on DataFrame
result = df.sql("""
    SELECT name, age, salary * 1.1 as new_salary
    FROM self
    WHERE age > 25
    ORDER BY salary DESC
""")

# Global SQL context
result = pl.sql("""
    SELECT df1.name, df2.department
    FROM df1
    JOIN df2 ON df1.id = df2.user_id
""", eager=True)
```

### Table Functions in SQL
```python
# Read files directly in SQL
result = pl.sql("""
    SELECT * FROM read_csv('data.csv')
    WHERE age > 25
""")
```

## Common Patterns

### Data Cleaning Pipeline
```python
cleaned_df = (
    pl.scan_csv("raw_data.csv")
    .filter(pl.col("id").is_not_null())
    .with_columns([
        pl.col("name").str.strip_chars().str.to_title(),
        pl.col("age").cast(pl.Int32),
        pl.col("email").str.to_lowercase()
    ])
    .filter(pl.col("age").is_between(18, 100))
    .unique(subset=["id"])
    .collect()
)
```

### Time Series Analysis
```python
time_series_df = (
    df.sort("timestamp")
    .with_columns([
        pl.col("value").rolling_mean(window_size=7).alias("7day_avg"),
        pl.col("value").rolling_std(window_size=7).alias("7day_std"),
        pl.col("value").pct_change().alias("pct_change")
    ])
    .filter(pl.col("timestamp") > pl.date(2023, 1, 1))
)
```

### Business Metrics
```python
metrics = (
    sales_df
    .group_by(["region", "product"])
    .agg([
        pl.col("revenue").sum().alias("total_revenue"),
        pl.col("quantity").sum().alias("total_quantity"),
        pl.col("customer_id").n_unique().alias("unique_customers"),
        (pl.col("revenue") / pl.col("quantity")).mean().alias("avg_price")
    ])
    .with_columns(
        (pl.col("total_revenue") / pl.col("total_revenue").sum()).alias("revenue_share")
    )
    .sort("total_revenue", descending=True)
)
```

## Error Handling & Debugging

### Common Errors
```python
# Schema mismatch in joins
try:
    result = df1.join(df2, on="id")
except pl.SchemaError as e:
    print(f"Schema error: {e}")

# Type casting errors
df = df.with_columns(
    pl.col("maybe_numeric").cast(pl.Float64, strict=False)
)
```

### Debugging
```python
# Inspect intermediate results
df.glimpse()  # Overview of DataFrame
df.describe()  # Statistical summary

# Check lazy query plan
lazy_df.explain()
lazy_df.show_graph()  # Visual representation
```

## Best Practices

1. **Use lazy evaluation** for large datasets and complex queries
2. **Project and filter early** to reduce memory usage
3. **Specify data types** explicitly when reading data
4. **Use categorical types** for string columns with limited unique values
5. **Leverage predicate pushdown** with file formats like Parquet
6. **Chain operations** in a single expression when possible
7. **Use window functions** instead of self-joins for analytical queries
8. **Prefer scan_* functions** over read_* for large files
9. **Use streaming** for datasets larger than memory
10. **Consider partitioning** for very large datasets

## Common Gotchas

- **Column selection**: `pl.col("name")` vs `"name"` - use expressions for consistency
- **Lazy evaluation**: Remember to `.collect()` lazy frames
- **Boolean operators**: Use `&` and `|` instead of `and` and `or`
- **String operations**: Chain `.str` methods properly
- **Join suffixes**: Specify suffixes to avoid column name conflicts
- **Memory usage**: Large eager DataFrames can consume significant memory
- **Type inference**: Be explicit with data types for predictable behavior

This cheat sheet covers the most commonly used Polars operations. For advanced features and detailed documentation, visit the official Polars documentation.