# Python

## Installation
```bash
# Download from python.org or use package manager
# macOS with Homebrew
brew install python

# Ubuntu/Debian
sudo apt update && sudo apt install python3 python3-pip

# Windows
# Download from python.org or use Microsoft Store

# Check version
python --version
python3 --version
```

## Basic Syntax

### Variables and Assignment
```python
# Variable assignment
x = 5
name = "Alice"
is_active = True

# Multiple assignment
a, b, c = 1, 2, 3
x = y = z = 0

# Swapping variables
a, b = b, a
```

### Data Types
```python
# Numeric types
integer = 42
float_num = 3.14
complex_num = 2 + 3j

# Strings
single_quote = 'Hello'
double_quote = "World"
multiline = """This is a
multiline string"""

# Boolean
is_true = True
is_false = False

# None type
empty_value = None

# Type checking
type(42)          # <class 'int'>
isinstance(42, int)  # True
```

## Operators

### Arithmetic Operators
```python
# Basic arithmetic
a + b    # Addition
a - b    # Subtraction
a * b    # Multiplication
a / b    # Division (float)
a // b   # Floor division
a % b    # Modulus
a ** b   # Exponentiation

# Augmented assignment
x += 5   # x = x + 5
x -= 3   # x = x - 3
x *= 2   # x = x * 2
x /= 4   # x = x / 4
```

### Comparison Operators
```python
a == b   # Equal
a != b   # Not equal
a < b    # Less than
a <= b   # Less than or equal
a > b    # Greater than
a >= b   # Greater than or equal

# Chaining comparisons
1 < x < 10  # True if x is between 1 and 10
```

### Logical Operators
```python
# Boolean logic
a and b  # Logical AND
a or b   # Logical OR
not a    # Logical NOT

# Short-circuit evaluation
result = condition1 and condition2  # condition2 only evaluated if condition1 is True
result = condition1 or condition2   # condition2 only evaluated if condition1 is False
```

### Identity and Membership
```python
# Identity operators
a is b       # Same object
a is not b   # Different objects

# Membership operators
item in container     # True if item is in container
item not in container # True if item is not in container

# Examples
x = [1, 2, 3]
2 in x      # True
4 not in x  # True
```

## Control Flow

### Conditional Statements
```python
# if-elif-else
if condition1:
    # code block
elif condition2:
    # code block
else:
    # code block

# Ternary operator
result = value_if_true if condition else value_if_false

# Examples
age = 18
status = "adult" if age >= 18 else "minor"

# Multiple conditions
if 18 <= age < 65:
    print("Working age")
```

### Loops

#### For Loops
```python
# Iterating over sequences
for item in [1, 2, 3]:
    print(item)

# Range function
for i in range(5):        # 0 to 4
    print(i)

for i in range(2, 8):     # 2 to 7
    print(i)

for i in range(0, 10, 2): # 0, 2, 4, 6, 8
    print(i)

# Enumerate for index and value
for index, value in enumerate(['a', 'b', 'c']):
    print(f"{index}: {value}")

# Zip for parallel iteration
for x, y in zip([1, 2, 3], ['a', 'b', 'c']):
    print(x, y)
```

#### While Loops
```python
# While loop
count = 0
while count < 5:
    print(count)
    count += 1

# Infinite loop with break
while True:
    user_input = input("Enter 'quit' to exit: ")
    if user_input == 'quit':
        break
    print(f"You entered: {user_input}")
```

#### Loop Control
```python
# break - exit loop
for i in range(10):
    if i == 5:
        break
    print(i)  # prints 0, 1, 2, 3, 4

# continue - skip iteration
for i in range(5):
    if i == 2:
        continue
    print(i)  # prints 0, 1, 3, 4

# else clause (executes if loop completes normally)
for i in range(3):
    print(i)
else:
    print("Loop completed")
```

## Data Structures

### Lists
```python
# Creating lists
empty_list = []
numbers = [1, 2, 3, 4, 5]
mixed = [1, "hello", 3.14, True]

# List methods
numbers.append(6)           # Add to end
numbers.insert(0, 0)        # Insert at index
numbers.extend([7, 8])      # Add multiple items
numbers.remove(3)           # Remove first occurrence
item = numbers.pop()        # Remove and return last item
item = numbers.pop(0)       # Remove and return item at index

# List operations
len(numbers)                # Length
numbers[0]                  # Access by index
numbers[-1]                 # Last item
numbers[1:4]                # Slicing
numbers[:3]                 # First 3 items
numbers[2:]                 # From index 2 to end
numbers[::2]                # Every 2nd item

# List comprehension
squares = [x**2 for x in range(10)]
evens = [x for x in range(20) if x % 2 == 0]
```

### Tuples
```python
# Creating tuples
empty_tuple = ()
single_item = (42,)         # Note the comma
coordinates = (10, 20)
rgb = (255, 128, 0)

# Tuple unpacking
x, y = coordinates
r, g, b = rgb

# Named tuples
from collections import namedtuple
Point = namedtuple('Point', ['x', 'y'])
p = Point(10, 20)
print(p.x, p.y)
```

### Dictionaries
```python
# Creating dictionaries
empty_dict = {}
person = {'name': 'Alice', 'age': 30, 'city': 'New York'}
person = dict(name='Alice', age=30, city='New York')

# Dictionary operations
person['name']              # Access value
person['email'] = 'alice@example.com'  # Add/update
del person['age']           # Delete key
person.get('phone', 'N/A')  # Get with default

# Dictionary methods
person.keys()               # All keys
person.values()             # All values
person.items()              # Key-value pairs
person.update({'age': 31})  # Update multiple

# Dictionary comprehension
squares = {x: x**2 for x in range(5)}
```

### Sets
```python
# Creating sets
empty_set = set()
numbers = {1, 2, 3, 4, 5}
from_list = set([1, 2, 2, 3])  # Duplicates removed

# Set operations
numbers.add(6)              # Add element
numbers.remove(3)           # Remove element (raises error if not found)
numbers.discard(10)         # Remove element (no error if not found)

# Set mathematics
set1 = {1, 2, 3}
set2 = {3, 4, 5}
set1 | set2                 # Union {1, 2, 3, 4, 5}
set1 & set2                 # Intersection {3}
set1 - set2                 # Difference {1, 2}
set1 ^ set2                 # Symmetric difference {1, 2, 4, 5}
```

## Functions

### Basic Functions
```python
# Function definition
def greet(name):
    return f"Hello, {name}!"

# Function with default parameters
def greet(name, greeting="Hello"):
    return f"{greeting}, {name}!"

# Multiple return values
def divide_and_remainder(a, b):
    return a // b, a % b

quotient, remainder = divide_and_remainder(17, 5)

# Variable arguments
def sum_all(*args):
    return sum(args)

def print_info(**kwargs):
    for key, value in kwargs.items():
        print(f"{key}: {value}")

# Mixed arguments
def complex_func(required, default="default", *args, **kwargs):
    print(f"Required: {required}")
    print(f"Default: {default}")
    print(f"Args: {args}")
    print(f"Kwargs: {kwargs}")
```

### Lambda Functions
```python
# Lambda (anonymous) functions
square = lambda x: x**2
add = lambda x, y: x + y

# Common use with higher-order functions
numbers = [1, 2, 3, 4, 5]
squares = list(map(lambda x: x**2, numbers))
evens = list(filter(lambda x: x % 2 == 0, numbers))

# Sorting with lambda
students = [('Alice', 85), ('Bob', 92), ('Charlie', 78)]
students.sort(key=lambda student: student[1])  # Sort by grade
```

### Decorators
```python
# Simple decorator
def my_decorator(func):
    def wrapper():
        print("Before function call")
        func()
        print("After function call")
    return wrapper

@my_decorator
def say_hello():
    print("Hello!")

# Decorator with arguments
def repeat(times):
    def decorator(func):
        def wrapper(*args, **kwargs):
            for _ in range(times):
                result = func(*args, **kwargs)
            return result
        return wrapper
    return decorator

@repeat(3)
def greet(name):
    print(f"Hello, {name}!")

# Built-in decorators
@staticmethod
def utility_function():
    pass

@classmethod
def from_string(cls, string):
    pass

@property
def full_name(self):
    return f"{self.first} {self.last}"
```

## Classes and Objects

### Basic Classes
```python
# Class definition
class Person:
    # Class variable
    species = "Homo sapiens"
    
    # Constructor
    def __init__(self, name, age):
        self.name = name    # Instance variable
        self.age = age
    
    # Instance method
    def greet(self):
        return f"Hello, I'm {self.name}"
    
    # String representation
    def __str__(self):
        return f"Person(name='{self.name}', age={self.age})"
    
    def __repr__(self):
        return f"Person('{self.name}', {self.age})"

# Creating objects
person1 = Person("Alice", 30)
print(person1.greet())
```

### Inheritance
```python
# Base class
class Animal:
    def __init__(self, name):
        self.name = name
    
    def speak(self):
        pass
    
    def __str__(self):
        return f"{self.__class__.__name__}('{self.name}')"

# Derived classes
class Dog(Animal):
    def speak(self):
        return f"{self.name} says Woof!"

class Cat(Animal):
    def speak(self):
        return f"{self.name} says Meow!"

# Multiple inheritance
class FlyingMixin:
    def fly(self):
        return f"{self.name} is flying!"

class Bird(Animal, FlyingMixin):
    def speak(self):
        return f"{self.name} says Tweet!"

# Using super()
class Employee(Person):
    def __init__(self, name, age, employee_id):
        super().__init__(name, age)
        self.employee_id = employee_id
```

### Special Methods (Magic Methods)
```python
class Vector:
    def __init__(self, x, y):
        self.x = x
        self.y = y
    
    def __add__(self, other):
        return Vector(self.x + other.x, self.y + other.y)
    
    def __sub__(self, other):
        return Vector(self.x - other.x, self.y - other.y)
    
    def __mul__(self, scalar):
        return Vector(self.x * scalar, self.y * scalar)
    
    def __len__(self):
        return (self.x**2 + self.y**2)**0.5  # Return magnitude as float
    
    def __eq__(self, other):
        return self.x == other.x and self.y == other.y
    
    def __str__(self):
        return f"Vector({self.x}, {self.y})"
```

## Exception Handling

### Basic Exception Handling
```python
# try-except
try:
    result = 10 / 0
except ZeroDivisionError:
    print("Cannot divide by zero")

# Multiple exceptions
try:
    value = int(input("Enter a number: "))
    result = 10 / value
except ValueError:
    print("Invalid input")
except ZeroDivisionError:
    print("Cannot divide by zero")

# Catch multiple exception types
try:
    # risky code
    pass
except (ValueError, TypeError) as e:
    print(f"Error: {e}")

# Catch all exceptions
try:
    # risky code
    pass
except Exception as e:
    print(f"Unexpected error: {e}")

# finally block (always executes)
try:
    file = open("data.txt")
    # process file
except FileNotFoundError:
    print("File not found")
finally:
    try:
        file.close()
    except NameError:
        pass  # file was never opened
```

### Custom Exceptions
```python
class CustomError(Exception):
    pass

class ValidationError(Exception):
    def __init__(self, message, code=None):
        super().__init__(message)
        self.code = code

def validate_age(age):
    if age < 0:
        raise ValidationError("Age cannot be negative", code="NEGATIVE_AGE")
    if age > 150:
        raise ValidationError("Age seems unrealistic", code="HIGH_AGE")

try:
    validate_age(-5)
except ValidationError as e:
    print(f"Validation failed: {e} (Code: {e.code})")
```

## File Handling

### Reading and Writing Files
```python
# Reading files
with open("file.txt", "r") as file:
    content = file.read()           # Read entire file
    
with open("file.txt", "r") as file:
    lines = file.readlines()        # Read all lines as list

with open("file.txt", "r") as file:
    for line in file:               # Iterate line by line
        print(line.strip())

# Writing files
with open("output.txt", "w") as file:
    file.write("Hello, World!\n")
    file.writelines(["Line 1\n", "Line 2\n"])

# Appending to files
with open("log.txt", "a") as file:
    file.write("New log entry\n")

# File modes
# "r" - read (default)
# "w" - write (overwrites)
# "a" - append
# "x" - exclusive creation
# "b" - binary mode
# "t" - text mode (default)
# "+" - read and write
```

### Working with JSON
```python
import json

# Writing JSON
data = {"name": "Alice", "age": 30, "city": "New York"}
with open("data.json", "w") as file:
    json.dump(data, file, indent=2)

# Reading JSON
with open("data.json", "r") as file:
    data = json.load(file)

# JSON strings
json_string = json.dumps(data, indent=2)
parsed_data = json.loads(json_string)
```

### Working with CSV
```python
import csv

# Writing CSV
data = [
    ["Name", "Age", "City"],
    ["Alice", 30, "New York"],
    ["Bob", 25, "Boston"]
]

with open("people.csv", "w", newline="") as file:
    writer = csv.writer(file)
    writer.writerows(data)

# Reading CSV
with open("people.csv", "r") as file:
    reader = csv.reader(file)
    for row in reader:
        print(row)

# CSV with dictionaries
with open("people.csv", "w", newline="") as file:
    fieldnames = ["name", "age", "city"]
    writer = csv.DictWriter(file, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerow({"name": "Alice", "age": 30, "city": "New York"})
```

## Modules and Packages

### Importing Modules
```python
# Different import styles
import math
from math import pi, sqrt
from math import *  # Not recommended
import math as m

# Using imported functions
result = math.sqrt(16)
result = sqrt(16)
result = m.sqrt(16)

# Importing from packages
from os.path import join, exists
from urllib.parse import urlparse
```

### Creating Modules
```python
# mymodule.py
"""A simple module example."""

PI = 3.14159

def circle_area(radius):
    """Calculate the area of a circle."""
    return PI * radius ** 2

def circle_circumference(radius):
    """Calculate the circumference of a circle."""
    return 2 * PI * radius

if __name__ == "__main__":
    # This code runs only when the module is executed directly
    print("Testing the module")
    print(f"Area of circle with radius 5: {circle_area(5)}")
```

### Package Structure
```
mypackage/
    __init__.py
    module1.py
    module2.py
    subpackage/
        __init__.py
        submodule.py
```

```python
# __init__.py
"""MyPackage - A sample package."""
from .module1 import function1
from .module2 import Class2

__version__ = "1.0.0"
__all__ = ["function1", "Class2"]
```

## Built-in Functions

### Common Built-ins
```python
# Type conversion
int("42")           # String to integer
float("3.14")       # String to float
str(42)             # Number to string
bool(1)             # To boolean
list("hello")       # String to list of characters

# Math functions
abs(-5)             # Absolute value
round(3.7)          # Round to nearest integer
round(3.14159, 2)   # Round to 2 decimal places
min(1, 2, 3)        # Minimum value
max([1, 2, 3])      # Maximum value
sum([1, 2, 3])      # Sum of iterable

# Sequence functions
len([1, 2, 3])      # Length
sorted([3, 1, 2])   # Return sorted list
reversed([1, 2, 3]) # Return reversed iterator
enumerate(['a', 'b']) # Return index-value pairs
zip([1, 2], ['a', 'b']) # Combine iterables

# Higher-order functions
map(str, [1, 2, 3])     # Apply function to each item
filter(lambda x: x > 0, [-1, 0, 1, 2])  # Filter items
all([True, True, False]) # True if all elements are true
any([False, True, False]) # True if any element is true
```

### Input/Output
```python
# User input
name = input("Enter your name: ")
age = int(input("Enter your age: "))

# Printing
print("Hello, World!")
print("Name:", name, "Age:", age)
print(f"Hello, {name}! You are {age} years old.")
print("Value:", 42, sep=" = ", end="\n\n")

# String formatting
name = "Alice"
age = 30
print("Name: {}, Age: {}".format(name, age))
print("Name: {name}, Age: {age}".format(name=name, age=age))
print(f"Name: {name}, Age: {age}")  # f-strings (Python 3.6+)
```

## String Operations

### String Methods
```python
text = "  Hello, World!  "

# Case conversion
text.upper()        # "  HELLO, WORLD!  "
text.lower()        # "  hello, world!  "
text.title()        # "  Hello, World!  "
text.capitalize()   # "  hello, world!  "
text.swapcase()     # "  hELLO, wORLD!  "

# Whitespace handling
text.strip()        # "Hello, World!"
text.lstrip()       # "Hello, World!  "
text.rstrip()       # "  Hello, World!"

# Searching and checking
text.find("World")      # Index of substring (or -1)
text.index("World")     # Index of substring (raises ValueError)
text.count("l")         # Count occurrences
text.startswith("  H")  # True
text.endswith("!  ")    # True
"123".isdigit()         # True
"abc".isalpha()         # True
"abc123".isalnum()      # True

# Splitting and joining
"a,b,c".split(",")      # ["a", "b", "c"]
"hello world".split()   # ["hello", "world"]
"-".join(["a", "b", "c"])  # "a-b-c"

# Replacement
text.replace("World", "Python")  # "  Hello, Python!  "

# Formatting
"The value is {:.2f}".format(3.14159)  # "The value is 3.14"
f"The value is {3.14159:.2f}"          # "The value is 3.14"
```

### String Slicing and Indexing
```python
text = "Hello, World!"

# Indexing
text[0]     # "H"
text[-1]    # "!"
text[7]     # "W"

# Slicing
text[0:5]   # "Hello"
text[7:]    # "World!"
text[:5]    # "Hello"
text[::2]   # "Hlo ol!"
text[::-1]  # "!dlroW ,olleH" (reversed)
```

## List Comprehensions and Generators

### List Comprehensions
```python
# Basic list comprehension
squares = [x**2 for x in range(10)]

# With condition
evens = [x for x in range(20) if x % 2 == 0]

# Nested loops
pairs = [(x, y) for x in range(3) for y in range(3) if x != y]

# Processing strings
words = ["hello", "world", "python"]
lengths = [len(word) for word in words]
upper_words = [word.upper() for word in words if len(word) > 4]
```

### Dictionary and Set Comprehensions
```python
# Dictionary comprehension
squares = {x: x**2 for x in range(5)}
word_lengths = {word: len(word) for word in ["hello", "world"]}

# Set comprehension
unique_lengths = {len(word) for word in ["hello", "world", "hi"]}
```

### Generator Expressions
```python
# Generator expression (lazy evaluation)
squares_gen = (x**2 for x in range(10))

# Generator function
def fibonacci(n):
    a, b = 0, 1
    for _ in range(n):
        yield a
        a, b = b, a + b

# Using generators
for num in fibonacci(10):
    print(num)

# Generator with yield from
def flatten(nested_list):
    for sublist in nested_list:
        if isinstance(sublist, list):
            yield from flatten(sublist)
        else:
            yield sublist

nested = [[1, 2], [3, [4, 5]], 6]
flat = list(flatten(nested))  # [1, 2, 3, 4, 5, 6]
```

## Context Managers

### Using Context Managers
```python
# File handling with context manager
with open("file.txt", "r") as file:
    content = file.read()
# File is automatically closed

# Multiple context managers
with open("input.txt", "r") as infile, open("output.txt", "w") as outfile:
    outfile.write(infile.read())
```

### Creating Context Managers
```python
# Using contextlib
from contextlib import contextmanager

@contextmanager
def timer():
    import time
    start = time.time()
    try:
        yield
    finally:
        end = time.time()
        print(f"Elapsed time: {end - start:.2f} seconds")

# Usage
with timer():
    # Some time-consuming operation
    sum(range(1000000))

# Class-based context manager
class DatabaseConnection:
    def __enter__(self):
        print("Connecting to database")
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        print("Closing database connection")
        return False  # Don't suppress exceptions

with DatabaseConnection() as db:
    print("Using database connection")
```

## Regular Expressions

### Basic Regex Operations
```python
import re

# Pattern matching
pattern = r"\d+"  # One or more digits
text = "I have 5 apples and 10 oranges"

# Search for first match
match = re.search(pattern, text)
if match:
    print(match.group())  # "5"

# Find all matches
matches = re.findall(pattern, text)  # ["5", "10"]

# Match at beginning of string
match = re.match(r"\w+", "Hello World")  # "Hello"

# Split by pattern
parts = re.split(r"\s+", "Hello    World   Python")  # ["Hello", "World", "Python"]

# Replace patterns
result = re.sub(r"\d+", "X", text)  # "I have X apples and X oranges"
```

### Common Regex Patterns
```python
# Email validation (simplified)
email_pattern = r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}"

# Phone number (US format)
phone_pattern = r"\(\d{3}\)\s\d{3}-\d{4}"

# URL pattern
url_pattern = r"https?://[^\s]+"

# Date pattern (YYYY-MM-DD)
date_pattern = r"\d{4}-\d{2}-\d{2}"

# Compiled patterns for efficiency
compiled_email = re.compile(email_pattern)
result = compiled_email.search("Contact us at info@example.com")
```

## Working with Dates and Times

### datetime Module
```python
from datetime import datetime, date, time, timedelta

# Current date and time
now = datetime.now()
today = date.today()
current_time = datetime.now().time()

# Creating specific dates
birthday = date(1990, 5, 15)
meeting_time = datetime(2023, 12, 25, 14, 30)

# Formatting dates
formatted = now.strftime("%Y-%m-%d %H:%M:%S")
formatted_date = today.strftime("%B %d, %Y")

# Parsing strings to dates
parsed = datetime.strptime("2023-12-25 14:30", "%Y-%m-%d %H:%M")

# Date arithmetic
tomorrow = today + timedelta(days=1)
week_ago = now - timedelta(weeks=1)
in_2_hours = now + timedelta(hours=2)

# Date comparisons
if birthday < today:
    print("Birthday has passed this year")
```

### time Module
```python
import time

# Current timestamp
timestamp = time.time()

# Sleep/pause execution
time.sleep(2)  # Pause for 2 seconds

# Measure execution time
start_time = time.time()
# ... some operation ...
end_time = time.time()
elapsed = end_time - start_time
```

## Error Handling Best Practices

### Specific Exception Handling
```python
def safe_divide(a, b):
    try:
        return a / b
    except TypeError:
        print("Both arguments must be numbers")
        return None
    except ZeroDivisionError:
        print("Cannot divide by zero")
        return None
    except Exception as e:
        print(f"Unexpected error: {e}")
        return None

def read_config_file(filename):
    try:
        with open(filename, 'r') as file:
            return json.load(file)
    except FileNotFoundError:
        print(f"Config file {filename} not found")
        return {}
    except json.JSONDecodeError as e:
        print(f"Invalid JSON in config file: {e}")
        return {}
    except PermissionError:
        print(f"Permission denied reading {filename}")
        return {}
```

## Common Patterns

### Singleton Pattern
```python
class Singleton:
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
```

### Factory Pattern
```python
class Animal:
    def speak(self):
        pass

class Dog(Animal):
    def speak(self):
        return "Woof!"

class Cat(Animal):
    def speak(self):
        return "Meow!"

def create_animal(animal_type):
    animals = {
        'dog': Dog,
        'cat': Cat
    }
    return animals.get(animal_type.lower(), Animal)()
```

## Performance Tips

1. **Use built-in functions**: `sum()`, `max()`, `min()` are optimized
2. **List comprehensions**: Generally faster than equivalent loops
3. **Use `join()` for string concatenation**: Instead of `+=` in loops
4. **Use `set` for membership testing**: O(1) vs O(n) for lists
5. **Use `collections.defaultdict`**: Avoid checking if keys exist
6. **Use generators**: For memory-efficient iteration over large datasets
7. **Profile your code**: Use `cProfile` and `timeit`

## Type Hints and Annotations

### Basic Type Hints
```python
from typing import List, Dict, Optional, Union, Callable, Tuple

# Function with type hints
def greet(name: str, age: int) -> str:
    return f"Hello {name}, you are {age} years old"

# Variable annotations
count: int = 0
names: List[str] = ["Alice", "Bob"]
scores: Dict[str, float] = {"Alice": 95.5, "Bob": 87.2}

# Optional and Union types
def find_user(user_id: int) -> Optional[str]:
    # Returns string or None
    return users.get(user_id)

def process_data(data: Union[str, int]) -> str:
    # Accepts string or int
    return str(data).upper()

# Function type hints
def apply_operation(func: Callable[[int, int], int], x: int, y: int) -> int:
    return func(x, y)

# Generic types
from typing import TypeVar
T = TypeVar('T')

def first_item(items: List[T]) -> Optional[T]:
    return items[0] if items else None
```

### Advanced Type Hints
```python
from typing import Protocol, Literal, Final

# Protocol for structural typing
class Drawable(Protocol):
    def draw(self) -> None: ...

# Literal types
def set_mode(mode: Literal['read', 'write', 'append']) -> None:
    pass

# Final variables (constants)
API_KEY: Final[str] = "secret-key"

# Class with type hints
class User:
    def __init__(self, name: str, email: str) -> None:
        self.name = name
        self.email = email
    
    def get_info(self) -> Dict[str, str]:
        return {"name": self.name, "email": self.email}
```

## Dataclasses

### Basic Dataclasses
```python
from dataclasses import dataclass, field
from typing import List

@dataclass
class Person:
    name: str
    age: int
    email: str = ""  # Default value
    
    def greet(self) -> str:
        return f"Hello, I'm {self.name}"

# Usage
person = Person("Alice", 30, "alice@example.com")
print(person.name)  # Alice

# With post-init processing
@dataclass
class Rectangle:
    width: float
    height: float
    area: float = field(init=False)  # Computed field
    
    def __post_init__(self):
        self.area = self.width * self.height

# Frozen dataclass (immutable)
@dataclass(frozen=True)
class Point:
    x: float
    y: float

# With default factory
@dataclass
class Team:
    name: str
    members: List[str] = field(default_factory=list)
```

## Async Programming

### Basic Async/Await
```python
import asyncio
import aiohttp
from typing import List

# Async function
async def fetch_data(url: str) -> str:
    async with aiohttp.ClientSession() as session:
        async with session.get(url) as response:
            return await response.text()

# Running async code
async def main():
    data = await fetch_data("https://api.example.com")
    print(data)

# Run the main function
asyncio.run(main())

# Concurrent execution
async def fetch_multiple(urls: List[str]) -> List[str]:
    tasks = [fetch_data(url) for url in urls]
    results = await asyncio.gather(*tasks)
    return results

# Async generator
async def async_counter(max_count: int):
    for i in range(max_count):
        yield i
        await asyncio.sleep(0.1)  # Simulate async work

# Using async generator
async def use_async_generator():
    async for number in async_counter(5):
        print(number)
```

## Collections Module

### Specialized Data Structures
```python
from collections import defaultdict, Counter, deque, namedtuple, OrderedDict

# defaultdict - provides default values for missing keys
dd = defaultdict(list)
dd['fruits'].append('apple')  # No KeyError
dd['fruits'].append('banana')

dd_int = defaultdict(int)
dd_int['count'] += 1  # Starts at 0

# Counter - count occurrences
text = "hello world"
counter = Counter(text)
print(counter['l'])  # 3
print(counter.most_common(3))  # [('l', 3), ('o', 2), ...]

# Count items in list
items = ['apple', 'banana', 'apple', 'orange', 'banana', 'apple']
item_counts = Counter(items)

# deque - double-ended queue (efficient append/pop from both ends)
dq = deque([1, 2, 3])
dq.appendleft(0)     # [0, 1, 2, 3]
dq.append(4)         # [0, 1, 2, 3, 4]
dq.popleft()         # [1, 2, 3, 4]
dq.pop()             # [1, 2, 3]

# Rotating
dq.rotate(1)         # [3, 1, 2]
dq.rotate(-1)        # [1, 2, 3]

# namedtuple - immutable classes
Point = namedtuple('Point', ['x', 'y'])
p = Point(10, 20)
print(p.x, p.y)      # 10 20
```

## Itertools

### Advanced Iteration Patterns
```python
import itertools

# Infinite iterators
counter = itertools.count(10, 2)  # 10, 12, 14, 16, ...
cycle_data = itertools.cycle(['A', 'B', 'C'])  # A, B, C, A, B, C, ...
repeated = itertools.repeat('hello', 3)  # 'hello', 'hello', 'hello'

# Finite iterators
data = [1, 2, 3, 4, 5]

# Chain multiple iterables
chained = itertools.chain([1, 2], [3, 4], [5, 6])  # 1, 2, 3, 4, 5, 6

# Combinations and permutations
combos = list(itertools.combinations([1, 2, 3], 2))  # [(1, 2), (1, 3), (2, 3)]
perms = list(itertools.permutations([1, 2, 3], 2))   # [(1, 2), (1, 3), (2, 1), ...]

# Groupby
data = [('a', 1), ('a', 2), ('b', 3), ('b', 4), ('c', 5)]
for key, group in itertools.groupby(data, key=lambda x: x[0]):
    print(f"{key}: {list(group)}")

# Take while condition is true
numbers = [1, 3, 5, 8, 9, 11, 13]
odds = list(itertools.takewhile(lambda x: x % 2 == 1, numbers))  # [1, 3, 5]

# Drop while condition is true
rest = list(itertools.dropwhile(lambda x: x % 2 == 1, numbers))  # [8, 9, 11, 13]

# Product (cartesian product)
colors = ['red', 'blue']
sizes = ['S', 'M', 'L']
variants = list(itertools.product(colors, sizes))
# [('red', 'S'), ('red', 'M'), ('red', 'L'), ('blue', 'S'), ...]
```

## Pathlib (Modern File Handling)

### Path Operations
```python
from pathlib import Path
import os

# Creating paths
current_dir = Path.cwd()
home_dir = Path.home()
file_path = Path("data/file.txt")
absolute_path = Path("/usr/local/bin/python")

# Path manipulation
file_path = Path("documents/projects/my_project/data.txt")
print(file_path.parent)        # documents/projects/my_project
print(file_path.name)          # data.txt
print(file_path.stem)          # data
print(file_path.suffix)        # .txt
print(file_path.parts)         # ('documents', 'projects', 'my_project', 'data.txt')

# Joining paths
project_dir = Path("projects")
config_file = project_dir / "config" / "settings.json"

# File operations
if file_path.exists():
    content = file_path.read_text()
    lines = file_path.read_text().splitlines()

# Write to file
output_path = Path("output.txt")
output_path.write_text("Hello, World!")

# Directory operations
data_dir = Path("data")
data_dir.mkdir(parents=True, exist_ok=True)  # Create directory

# List directory contents
for item in data_dir.iterdir():
    if item.is_file():
        print(f"File: {item.name}")
    elif item.is_dir():
        print(f"Directory: {item.name}")

# Glob patterns
python_files = list(Path(".").glob("*.py"))
all_python_files = list(Path(".").rglob("*.py"))  # Recursive
```

## Logging

### Basic Logging
```python
import logging

# Basic configuration
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('app.log'),
        logging.StreamHandler()
    ]
)

# Create logger
logger = logging.getLogger(__name__)

# Log levels
logger.debug("Debug message")     # Not shown with INFO level
logger.info("Info message")
logger.warning("Warning message")
logger.error("Error message")
logger.critical("Critical message")

# Logging with variables
name = "Alice"
age = 30
logger.info(f"User {name} is {age} years old")

# Exception logging
try:
    result = 10 / 0
except ZeroDivisionError:
    logger.exception("Division by zero occurred")  # Includes traceback
```

### Advanced Logging
```python
import logging.config

# Dictionary configuration
LOGGING_CONFIG = {
    'version': 1,
    'disable_existing_loggers': False,
    'formatters': {
        'standard': {
            'format': '%(asctime)s [%(levelname)s] %(name)s: %(message)s'
        },
    },
    'handlers': {
        'file': {
            'level': 'INFO',
            'class': 'logging.handlers.RotatingFileHandler',
            'filename': 'app.log',
            'maxBytes': 1024*1024*5,  # 5MB
            'backupCount': 3,
            'formatter': 'standard',
        },
        'console': {
            'level': 'DEBUG',
            'class': 'logging.StreamHandler',
            'formatter': 'standard',
        },
    },
    'root': {
        'handlers': ['file', 'console'],
        'level': 'INFO',
    }
}

logging.config.dictConfig(LOGGING_CONFIG)
logger = logging.getLogger(__name__)
```

## Command Line Arguments

### Using argparse
```python
import argparse

def main():
    parser = argparse.ArgumentParser(description='Process some files.')
    
    # Positional arguments
    parser.add_argument('filename', help='Input filename')
    
    # Optional arguments
    parser.add_argument('-o', '--output', help='Output filename')
    parser.add_argument('-v', '--verbose', action='store_true', help='Verbose output')
    parser.add_argument('--count', type=int, default=1, help='Number of times to process')
    parser.add_argument('--format', choices=['json', 'csv', 'xml'], default='json')
    
    # Parse arguments
    args = parser.parse_args()
    
    print(f"Input file: {args.filename}")
    if args.output:
        print(f"Output file: {args.output}")
    if args.verbose:
        print("Verbose mode enabled")
    print(f"Processing {args.count} times in {args.format} format")

if __name__ == "__main__":
    main()

# Usage examples:
# python script.py input.txt
# python script.py input.txt -o output.txt --verbose --count 3 --format csv
```

## Testing

### Using unittest
```python
import unittest
from mymodule import Calculator

class TestCalculator(unittest.TestCase):
    
    def setUp(self):
        """Set up test fixtures before each test method."""
        self.calc = Calculator()
    
    def test_add(self):
        """Test addition method."""
        result = self.calc.add(2, 3)
        self.assertEqual(result, 5)
    
    def test_divide_by_zero(self):
        """Test division by zero raises exception."""
        with self.assertRaises(ValueError):
            self.calc.divide(10, 0)
    
    def test_multiple_operations(self):
        """Test multiple operations."""
        self.assertEqual(self.calc.add(2, 3), 5)
        self.assertEqual(self.calc.subtract(10, 4), 6)
        self.assertEqual(self.calc.multiply(3, 7), 21)
    
    def tearDown(self):
        """Clean up after each test method."""
        pass

# Run tests
if __name__ == '__main__':
    unittest.main()

# Common assertions
# self.assertEqual(a, b)      # a == b
# self.assertNotEqual(a, b)   # a != b
# self.assertTrue(x)          # bool(x) is True
# self.assertFalse(x)         # bool(x) is False
# self.assertIs(a, b)         # a is b
# self.assertIsNot(a, b)      # a is not b
# self.assertIsNone(x)        # x is None
# self.assertIn(a, b)         # a in b
# self.assertGreater(a, b)    # a > b
# self.assertRaises(exc, fun, *args)
```

### Using pytest (Alternative)
```python
import pytest
from mymodule import Calculator

# Fixture for test setup
@pytest.fixture
def calculator():
    return Calculator()

def test_add(calculator):
    assert calculator.add(2, 3) == 5

def test_divide_by_zero(calculator):
    with pytest.raises(ValueError):
        calculator.divide(10, 0)

# Parametrized tests
@pytest.mark.parametrize("a,b,expected", [
    (2, 3, 5),
    (1, 1, 2),
    (-1, 1, 0),
    (0, 0, 0),
])
def test_add_parametrized(calculator, a, b, expected):
    assert calculator.add(a, b) == expected

# Run with: pytest test_file.py
```

## Virtual Environments and Package Management

### Creating Virtual Environments
```bash
# Using venv (built-in)
python -m venv myenv
source myenv/bin/activate  # Linux/Mac
myenv\Scripts\activate     # Windows
deactivate                 # Exit virtual environment

# Using virtualenv
pip install virtualenv
virtualenv myenv
source myenv/bin/activate

# Using conda
conda create --name myenv python=3.9
conda activate myenv
conda deactivate
```

### Package Management
```bash
# Install packages
pip install requests
pip install requests==2.28.0  # Specific version
pip install requests>=2.25.0  # Minimum version

# Install from requirements file
pip install -r requirements.txt

# Create requirements file
pip freeze > requirements.txt

# Install in development mode
pip install -e .

# Upgrade packages
pip install --upgrade requests
pip list --outdated
```

### requirements.txt example
```
requests==2.28.0
pandas>=1.4.0
numpy>=1.21.0,<2.0.0
pytest>=7.0.0
black==22.6.0
```

## Common Third-Party Libraries

### Requests (HTTP Client)
```python
import requests
import json

# GET request
response = requests.get('https://api.github.com/users/octocat')
if response.status_code == 200:
    data = response.json()
    print(data['name'])

# POST request
payload = {'key1': 'value1', 'key2': 'value2'}
response = requests.post('https://httpbin.org/post', json=payload)

# With headers and authentication
headers = {'Authorization': 'Bearer token123'}
response = requests.get('https://api.example.com/data', headers=headers)

# Session for multiple requests
session = requests.Session()
session.headers.update({'User-Agent': 'MyApp/1.0'})
response = session.get('https://api.example.com')
```

### Basic Data Analysis (Pandas)
```python
import pandas as pd
import numpy as np

# Create DataFrame
data = {
    'name': ['Alice', 'Bob', 'Charlie'],
    'age': [25, 30, 35],
    'city': ['NYC', 'LA', 'Chicago']
}
df = pd.DataFrame(data)

# Read from files
df = pd.read_csv('data.csv')
df = pd.read_json('data.json')
df = pd.read_excel('data.xlsx')

# Basic operations
print(df.head())        # First 5 rows
print(df.info())        # Data info
print(df.describe())    # Statistics

# Selecting data
ages = df['age']        # Single column
subset = df[['name', 'age']]  # Multiple columns
filtered = df[df['age'] > 25]  # Filter rows

# Save data
df.to_csv('output.csv', index=False)
df.to_json('output.json')
```

## Best Practices

1. **Follow PEP 8**: Python style guide
2. **Use meaningful variable names**: `user_count` not `uc`
3. **Write docstrings**: Document functions and classes
4. **Use type hints**: Help with code clarity and tools
5. **Handle exceptions gracefully**: Don't use bare `except:`
6. **Use context managers**: For resource management
7. **Keep functions small**: Single responsibility principle
8. **Use virtual environments**: Isolate project dependencies
9. **Test your code**: Write unit tests
10. **Use version control**: Git for tracking changes
11. **Use pathlib**: Instead of os.path for file operations
12. **Prefer f-strings**: For string formatting in Python 3.6+
13. **Use dataclasses**: For simple data containers
14. **Use async/await**: For I/O-bound operations
15. **Profile before optimizing**: Use cProfile, timeit, or py-spy