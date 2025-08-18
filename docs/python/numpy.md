# NumPy

## Installation
```bash
pip install numpy
```

## Import
```python
import numpy as np
```

## Array Creation

### Basic Creation
```python
# From lists
arr = np.array([1, 2, 3, 4])
arr_2d = np.array([[1, 2], [3, 4]])

# Zeros, ones, empty
np.zeros((3, 4))          # 3x4 array of zeros
np.ones((2, 3))           # 2x3 array of ones
np.empty((2, 2))          # uninitialized 2x2 array

# Identity matrix
np.eye(3)                 # 3x3 identity matrix
np.identity(4)            # 4x4 identity matrix
```

### Range Arrays
```python
np.arange(10)             # [0, 1, 2, ..., 9]
np.arange(2, 10, 2)       # [2, 4, 6, 8]
np.linspace(0, 1, 5)      # 5 evenly spaced values from 0 to 1
```

### Random Arrays
```python
np.random.random((3, 3))      # 3x3 random floats [0, 1)
np.random.randint(0, 10, 5)   # 5 random ints [0, 10)
np.random.normal(0, 1, 100)   # 100 samples from normal distribution
np.random.seed(42)            # set random seed
```

## Array Properties
```python
arr.shape              # dimensions
arr.size               # total number of elements
arr.ndim               # number of dimensions
arr.dtype              # data type
arr.itemsize           # size of each element in bytes
```

## Array Operations

### Basic Math
```python
arr + 5                # add scalar
arr * 2                # multiply by scalar
arr1 + arr2            # element-wise addition
arr1 * arr2            # element-wise multiplication
arr1 @ arr2            # matrix multiplication
np.dot(arr1, arr2)     # matrix multiplication (alternative)
```

### Mathematical Functions
```python
np.sqrt(arr)           # square root
np.exp(arr)            # exponential
np.log(arr)            # natural log
np.sin(arr), np.cos(arr), np.tan(arr)  # trig functions
np.abs(arr)            # absolute value
```

### Aggregation Functions
```python
np.sum(arr)            # sum all elements
np.mean(arr)           # mean
np.median(arr)         # median
np.std(arr)            # standard deviation
np.min(arr), np.max(arr)  # min/max
np.argmin(arr), np.argmax(arr)  # indices of min/max
```

### Axis Operations
```python
arr_2d.sum(axis=0)     # sum along rows (column-wise)
arr_2d.sum(axis=1)     # sum along columns (row-wise)
arr_2d.mean(axis=0)    # mean along axis 0
```

## Array Indexing and Slicing

### Basic Indexing
```python
arr[0]                 # first element
arr[-1]                # last element
arr[1:4]               # elements 1, 2, 3
arr[::2]               # every second element
```

### 2D Indexing
```python
arr_2d[0, 1]           # element at row 0, column 1
arr_2d[0, :]           # entire first row
arr_2d[:, 1]           # entire second column
arr_2d[1:3, 0:2]       # subarray
```

### Boolean Indexing
```python
mask = arr > 5
arr[mask]              # elements greater than 5
arr[arr > 5]           # same as above
arr[(arr > 5) & (arr < 10)]  # multiple conditions
```

### Fancy Indexing
```python
indices = [0, 2, 4]
arr[indices]           # elements at specified indices
arr_2d[[0, 2], [1, 3]] # elements at (0,1) and (2,3)
```

## Array Manipulation

### Reshaping
```python
arr.reshape(3, 4)      # reshape to 3x4
arr.flatten()          # flatten to 1D
arr.ravel()            # flatten to 1D (view if possible)
arr.T                  # transpose
np.transpose(arr)      # transpose (alternative)
```

### Joining Arrays
```python
np.concatenate([arr1, arr2])          # concatenate along existing axis
np.vstack([arr1, arr2])               # stack vertically
np.hstack([arr1, arr2])               # stack horizontally
np.column_stack([arr1, arr2])         # stack as columns
```

### Splitting Arrays
```python
np.split(arr, 3)       # split into 3 equal parts
np.hsplit(arr_2d, 2)   # split horizontally
np.vsplit(arr_2d, 2)   # split vertically
```

## Advanced Operations

### Broadcasting
```python
# Arrays of different shapes can be operated on
arr_2d = np.array([[1, 2, 3], [4, 5, 6]])
arr_1d = np.array([10, 20, 30])
result = arr_2d + arr_1d  # broadcasts arr_1d across rows
```

### Conditional Operations
```python
np.where(arr > 5, arr, 0)     # replace values <= 5 with 0
np.select([arr < 5, arr > 10], [0, 100], arr)  # multiple conditions
```

### Unique and Set Operations
```python
np.unique(arr)                # unique elements
np.in1d(arr1, arr2)          # test membership
np.intersect1d(arr1, arr2)   # intersection
np.union1d(arr1, arr2)       # union
```

### Sorting
```python
np.sort(arr)              # sort array
np.argsort(arr)           # indices that would sort array
np.partition(arr, 3)      # partition around 3rd element
```

## Linear Algebra
```python
# Dot product
np.dot(a, b)

# Matrix multiplication
a @ b

# Eigenvalues and eigenvectors
np.linalg.eig(matrix)

# Singular value decomposition
np.linalg.svd(matrix)

# Matrix inverse
np.linalg.inv(matrix)

# Determinant
np.linalg.det(matrix)

# Solve linear system Ax = b
np.linalg.solve(A, b)
```

## Performance Tips
- Use vectorized operations instead of loops
- Avoid unnecessary array copies
- Use views when possible (`arr.view()`)
- Pre-allocate arrays when size is known
- Use appropriate data types (`dtype`) to save memory