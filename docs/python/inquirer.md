# Python Inquirer Cheat Sheet

A comprehensive reference for building interactive command-line interfaces with Python's inquirer libraries. This covers both the original `python-inquirer` and the modern `InquirerPy` libraries for creating engaging CLI prompts.

## Quick Start

### Installation
```bash
# Original python-inquirer (stable, simple)
pip install inquirer

# InquirerPy (modern, feature-rich)
pip install inquirerpy

# Both libraries can coexist
pip install inquirer inquirerpy
```

### Basic Usage Comparison
```python
# python-inquirer (classic)
import inquirer
questions = [
    inquirer.Text('name', message="What's your name?"),
    inquirer.List('color', message="Favorite color?", choices=['Red', 'Blue', 'Green'])
]
answers = inquirer.prompt(questions)

# InquirerPy (modern - classic syntax)
from InquirerPy import prompt
questions = [
    {"type": "input", "message": "What's your name?", "name": "name"},
    {"type": "list", "message": "Favorite color?", "choices": ["Red", "Blue", "Green"]}
]
answers = prompt(questions)

# InquirerPy (modern - alternate syntax)
from InquirerPy import inquirer
name = inquirer.text(message="What's your name?").execute()
color = inquirer.select(message="Favorite color?", choices=["Red", "Blue", "Green"]).execute()
```

## Core Prompt Types

### Text Input

#### python-inquirer
```python
import inquirer

# Basic text input
questions = [
    inquirer.Text('username', message="Enter username"),
    inquirer.Text('email', message="Enter email", 
                  validate=lambda _, x: '@' in x),
]
answers = inquirer.prompt(questions)
```

#### InquirerPy
```python
# Classic syntax
from InquirerPy import prompt
questions = [
    {
        "type": "input",
        "message": "Username:",
        "name": "username",
        "validate": lambda result: len(result) > 0,
        "invalid_message": "Username cannot be empty"
    }
]
result = prompt(questions)

# Alternate syntax
from InquirerPy import inquirer
username = inquirer.text(
    message="Username:",
    validate=lambda result: len(result) > 0,
    invalid_message="Username cannot be empty"
).execute()
```

### Password Input

#### python-inquirer
```python
import inquirer

questions = [
    inquirer.Password('password', message="Enter password")
]
answers = inquirer.prompt(questions)
```

#### InquirerPy
```python
# Classic syntax
from InquirerPy import prompt
from InquirerPy.validator import PasswordValidator

questions = [
    {
        "type": "secret",
        "message": "Password:",
        "validate": PasswordValidator(
            length=8,
            cap=True,
            special=True,
            number=True
        )
    }
]
result = prompt(questions)

# Alternate syntax
password = inquirer.secret(
    message="Password:",
    validate=PasswordValidator(length=8, cap=True, special=True, number=True)
).execute()
```

### Single Choice Lists

#### python-inquirer
```python
import inquirer

questions = [
    inquirer.List('size',
                  message="What size?",
                  choices=['Small', 'Medium', 'Large'],
                  carousel=True  # Circular navigation
                  )
]
answers = inquirer.prompt(questions)
```

#### InquirerPy
```python
# Classic syntax
questions = [
    {
        "type": "list",
        "message": "What size?",
        "choices": ["Small", "Medium", "Large"],
        "default": "Medium"
    }
]
result = prompt(questions)

# Alternate syntax
size = inquirer.select(
    message="What size?",
    choices=["Small", "Medium", "Large"],
    default="Medium"
).execute()
```

### Multiple Choice (Checkboxes)

#### python-inquirer
```python
import inquirer

questions = [
    inquirer.Checkbox('interests',
                      message="Select interests",
                      choices=['Music', 'Sports', 'Reading', 'Gaming'],
                      )
]
answers = inquirer.prompt(questions)
```

#### InquirerPy
```python
# With validation for minimum selections
from InquirerPy import inquirer

interests = inquirer.checkbox(
    message="Select interests:",
    choices=["Music", "Sports", "Reading", "Gaming"],
    validate=lambda selection: len(selection) >= 2,
    invalid_message="Select at least 2 interests"
).execute()
```

### Confirmation Prompts

#### python-inquirer
```python
import inquirer

questions = [
    inquirer.Confirm('proceed', message="Continue?", default=True)
]
answers = inquirer.prompt(questions)
```

#### InquirerPy
```python
# Custom confirmation letters (localization)
from InquirerPy import inquirer

confirm = inquirer.confirm(
    message="Proceed?",
    default=True,
    confirm_letter="s",  # 's' for 'Sim' (Yes in Portuguese)
    reject_letter="n",   # 'n' for 'Não' (No in Portuguese)
    transformer=lambda result: "Sim" if result else "Não"
).execute()
```

### File Path Selection

#### python-inquirer
```python
import inquirer

questions = [
    inquirer.Path('config_file',
                  message="Config file location?",
                  path_type=inquirer.Path.FILE,
                  exists=True
                  )
]
answers = inquirer.prompt(questions)
```

#### InquirerPy
```python
from InquirerPy import inquirer
from InquirerPy.validator import PathValidator

filepath = inquirer.filepath(
    message="Select file:",
    validate=PathValidator("Path must be valid")
).execute()
```

## Advanced Features

### Dynamic Questions (Conditional Logic)

#### python-inquirer
```python
import inquirer

questions = [
    inquirer.Confirm('married', message="Are you married?"),
    inquirer.Text('spouse_name',
                  message="Spouse name?",
                  ignore=lambda x: not x['married']  # Skip if not married
                  )
]
answers = inquirer.prompt(questions)
```

#### InquirerPy
```python
from InquirerPy import prompt

questions = [
    {"type": "confirm", "message": "Are you married?", "name": "married"},
    {
        "type": "input",
        "message": "Spouse name?",
        "name": "spouse_name",
        "when": lambda result: result["married"]  # Show only if married
    }
]
result = prompt(questions)
```

### Choice Objects and Separators

#### InquirerPy Advanced Choices
```python
from InquirerPy import inquirer
from InquirerPy.base.control import Choice
from InquirerPy.separator import Separator

# Advanced choice configuration
choices = [
    Choice("aws-east-1", name="AWS East (Virginia)", enabled=True),
    Choice("aws-west-1", name="AWS West (California)", enabled=False),
    Separator(),
    "gcp-us-central",
    "azure-eastus"
]

region = inquirer.select(
    message="Select cloud region:",
    choices=choices,
    multiselect=True,
    transformer=lambda result: f"{len(result)} region(s) selected"
).execute()
```

### Custom Validation

#### python-inquirer
```python
import inquirer
import re

def phone_validator(answers, current):
    if not re.match(r'^\+?\d[\d ]+\d$', current):
        raise inquirer.errors.ValidationError('', reason='Invalid phone format')
    return True

questions = [
    inquirer.Text('phone', 
                  message="Phone number",
                  validate=phone_validator)
]
answers = inquirer.prompt(questions)
```

#### InquirerPy
```python
from InquirerPy import inquirer
from InquirerPy.validator import NumberValidator, EmptyInputValidator
import re

# Built-in validators
age = inquirer.text(
    message="Age:",
    validate=NumberValidator(float_allowed=False),
    filter=lambda result: int(result)  # Convert to integer
).execute()

# Custom validator function
def email_validator(email):
    pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    return re.match(pattern, email) is not None

email = inquirer.text(
    message="Email:",
    validate=email_validator,
    invalid_message="Please enter a valid email address"
).execute()

# Custom validator class
from prompt_toolkit.validation import ValidationError, Validator

class CustomEmailValidator(Validator):
    def validate(self, document):
        if '@' not in document.text:
            raise ValidationError(
                message="Email must contain @ symbol",
                cursor_position=len(document.text)
            )

email = inquirer.text(
    message="Email:",
    validate=CustomEmailValidator()
).execute()
```

## Styling and Theming

### InquirerPy Styling
```python
from InquirerPy import prompt
from InquirerPy.utils import color_print

# Custom style dictionary
custom_style = {
    "questionmark": "#ff9d00 bold",
    "answer": "#61afef",
    "input": "#98c379",
    "question": "",
    "answered_question": "",
    "instruction": "#abb2bf",
    "pointer": "#61afef",
    "checkbox": "#98c379",
    "separator": "",
    "skipped": "#5c6370",
    "validator": "#e06c75",
    "marker": "#e5c07b",
}

result = prompt(
    {"type": "input", "message": "Styled prompt:"},
    style=custom_style,
    vi_mode=True  # Enable vim keybindings
)

# Environment variable styling
import os
os.environ["INQUIRERPY_STYLE_QUESTIONMARK"] = "#ff9d00 bold"
os.environ["INQUIRERPY_STYLE_ANSWER"] = "#61afef"

# Color printing utility
color_print([("#e5c07b", "Hello "), ("#61afef", "World!")])
```

### Default InquirerPy Theme
```python
# Based on onedark color palette
default_style = {
    "questionmark": "#e5c07b",      # Yellow
    "answermark": "#e5c07b",        # Yellow
    "answer": "#61afef",            # Blue
    "input": "#98c379",             # Green
    "question": "",                 # Default
    "answered_question": "",        # Default
    "instruction": "#abb2bf",       # Light gray
    "long_instruction": "#abb2bf",  # Light gray
    "pointer": "#61afef",           # Blue
    "checkbox": "#98c379",          # Green
    "separator": "",                # Default
    "skipped": "#5c6370",          # Dark gray
    "validator": "",                # Default
    "marker": "#e5c07b",           # Yellow
    "fuzzy_prompt": "#c678dd",     # Purple
    "fuzzy_info": "#abb2bf",       # Light gray
    "fuzzy_border": "#4b5263",     # Dark blue
    "fuzzy_match": "#c678dd",      # Purple
    "spinner_pattern": "#e5c07b",   # Yellow
    "spinner_text": "",            # Default
}
```

## Advanced InquirerPy Features

### Fuzzy Search
```python
from InquirerPy import inquirer

# Large list with fuzzy search
frameworks = [
    "React", "Vue", "Angular", "Svelte", "Next.js",
    "Django", "Flask", "FastAPI", "Express", "Koa",
    "Spring Boot", "Laravel", "Ruby on Rails"
]

framework = inquirer.fuzzy(
    message="Select framework:",
    choices=frameworks,
    max_height="70%",  # 70% of terminal height
    match_exact=True,  # Enable exact substring matching
    exact_symbol=" E"  # Indicator for exact matches
).execute()
```

### Expand Choices
```python
from InquirerPy import inquirer
from InquirerPy.prompts.expand import ExpandChoice, ExpandHelp

# Expand prompt for quick selection
choices = [
    ExpandChoice("create", key="c", name="Create new project"),
    ExpandChoice("open", key="o", name="Open existing project"),
    ExpandChoice("delete", key="d", name="Delete project"),
    ExpandChoice("quit", key="q", name="Quit application")
]

action = inquirer.expand(
    message="What would you like to do?",
    choices=choices,
    expand_help=ExpandHelp(key="h", message="Show help")
).execute()
```

### Number Input
```python
from InquirerPy import inquirer
from InquirerPy.validator import NumberValidator

# Number input with validation
age = inquirer.number(
    message="Enter your age:",
    min_allowed=0,
    max_allowed=150,
    validate=NumberValidator(),
    replace_mode=True  # Replace entire input on type
).execute()
```

### Custom Keybindings
```python
from InquirerPy import inquirer

# Custom keybindings
keybindings = {
    "skip": [{"key": "c-c"}],        # Ctrl+C to skip
    "interrupt": [{"key": "c-d"}],   # Ctrl+D to interrupt
    "toggle-all": [{"key": ["c-a", "space"]}]  # Ctrl+A then Space
}

result = inquirer.select(
    message="Select options:",
    choices=["Option 1", "Option 2", "Option 3", "Option 4"],
    multiselect=True,
    keybindings=keybindings,
    vi_mode=True  # Enable vim mode
).execute()
```

### Height Control
```python
from InquirerPy import inquirer

# Control prompt height
result = inquirer.select(
    message="Select from long list:",
    choices=[f"Item {i}" for i in range(100)],
    height=10,        # Fixed height of 10 lines
    max_height="50%", # Max 50% of terminal height
    instruction="Use j/k to navigate"
).execute()
```

## Practical Examples

### User Registration Form
```python
from InquirerPy import inquirer
from InquirerPy.validator import EmptyInputValidator, PasswordValidator
import re

def email_validator(email):
    pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    if not re.match(pattern, email):
        return False
    return True

def register_user():
    print("=== User Registration ===")
    
    # Collect user information
    username = inquirer.text(
        message="Username:",
        validate=EmptyInputValidator("Username is required"),
        instruction="Letters, numbers, and underscores only"
    ).execute()
    
    email = inquirer.text(
        message="Email address:",
        validate=email_validator,
        invalid_message="Please enter a valid email address"
    ).execute()
    
    password = inquirer.secret(
        message="Password:",
        validate=PasswordValidator(
            length=8,
            cap=True,
            special=True,
            number=True,
            message="Password must be 8+ chars with uppercase, number, and special character"
        )
    ).execute()
    
    # Confirm password
    confirm_password = inquirer.secret(
        message="Confirm password:",
        validate=lambda pwd: pwd == password,
        invalid_message="Passwords do not match"
    ).execute()
    
    # Additional preferences
    newsletter = inquirer.confirm(
        message="Subscribe to newsletter?",
        default=False
    ).execute()
    
    interests = inquirer.checkbox(
        message="Select interests:",
        choices=[
            "Technology", "Sports", "Music", "Travel",
            "Food", "Books", "Movies", "Gaming"
        ],
        validate=lambda selection: len(selection) > 0,
        invalid_message="Please select at least one interest"
    ).execute()
    
    # Summary
    print(f"\n✅ Registration successful!")
    print(f"Username: {username}")
    print(f"Email: {email}")
    print(f"Newsletter: {'Yes' if newsletter else 'No'}")
    print(f"Interests: {', '.join(interests)}")
    
    return {
        'username': username,
        'email': email,
        'password': password,
        'newsletter': newsletter,
        'interests': interests
    }

# Run registration
user_data = register_user()
```

### Project Setup Wizard
```python
from InquirerPy import inquirer
from InquirerPy.base.control import Choice
from InquirerPy.separator import Separator
import os

def project_setup():
    print("🚀 Project Setup Wizard")
    
    # Project type selection
    project_type = inquirer.select(
        message="What type of project?",
        choices=[
            Choice("web", name="🌐 Web Application"),
            Choice("api", name="🔗 REST API"),
            Choice("desktop", name="🖥️  Desktop Application"),
            Choice("cli", name="⚡ Command Line Tool"),
            Choice("library", name="📚 Library/Package")
        ]
    ).execute()
    
    # Programming language
    language = inquirer.select(
        message="Programming language:",
        choices=[
            "Python", "JavaScript", "TypeScript", "Java",
            "Go", "Rust", "C++", "C#"
        ]
    ).execute()
    
    # Framework selection (conditional)
    framework = None
    if project_type == "web":
        if language == "Python":
            framework = inquirer.select(
                message="Web framework:",
                choices=["Django", "Flask", "FastAPI", "Tornado"]
            ).execute()
        elif language in ["JavaScript", "TypeScript"]:
            framework = inquirer.select(
                message="Web framework:",
                choices=["React", "Vue", "Angular", "Next.js", "Express"]
            ).execute()
    
    # Features
    features = inquirer.checkbox(
        message="Select features to include:",
        choices=[
            "Database integration",
            "Authentication",
            "Testing setup",
            "Docker support",
            "CI/CD pipeline",
            "Documentation",
            "Logging",
            "Configuration management"
        ]
    ).execute()
    
    # Project name and location
    project_name = inquirer.text(
        message="Project name:",
        validate=lambda name: len(name) > 0 and name.replace('-', '').replace('_', '').isalnum(),
        invalid_message="Project name must contain only letters, numbers, hyphens, and underscores"
    ).execute()
    
    default_path = os.path.join(os.path.expanduser("~"), "projects", project_name)
    project_path = inquirer.text(
        message="Project location:",
        default=default_path
    ).execute()
    
    # Confirmation
    print(f"\n📋 Project Summary:")
    print(f"Type: {project_type}")
    print(f"Language: {language}")
    if framework:
        print(f"Framework: {framework}")
    print(f"Features: {', '.join(features)}")
    print(f"Name: {project_name}")
    print(f"Location: {project_path}")
    
    proceed = inquirer.confirm(
        message="Create project with these settings?",
        default=True
    ).execute()
    
    if proceed:
        print("✅ Project created successfully!")
        # Here you would create the actual project structure
        return {
            'type': project_type,
            'language': language,
            'framework': framework,
            'features': features,
            'name': project_name,
            'path': project_path
        }
    else:
        print("❌ Project creation cancelled.")
        return None

# Run project setup
project_config = project_setup()
```

### Configuration Manager
```python
from InquirerPy import inquirer
from InquirerPy.validator import NumberValidator
import json
import os

def manage_config():
    config_file = "app_config.json"
    
    # Load existing config
    if os.path.exists(config_file):
        with open(config_file, 'r') as f:
            config = json.load(f)
        print("📂 Loaded existing configuration")
    else:
        config = {}
        print("🆕 Creating new configuration")
    
    # Menu system
    while True:
        action = inquirer.select(
            message="Configuration Manager:",
            choices=[
                Choice("view", name="👀 View current config"),
                Choice("edit", name="✏️  Edit settings"),
                Choice("add", name="➕ Add new setting"),
                Choice("delete", name="🗑️  Delete setting"),
                Choice("save", name="💾 Save and exit"),
                Choice("exit", name="🚪 Exit without saving")
            ]
        ).execute()
        
        if action == "view":
            if config:
                print("\n📋 Current Configuration:")
                for key, value in config.items():
                    print(f"  {key}: {value}")
            else:
                print("⚠️  Configuration is empty")
                
        elif action == "edit":
            if not config:
                print("⚠️  No settings to edit")
                continue
                
            setting = inquirer.select(
                message="Select setting to edit:",
                choices=list(config.keys())
            ).execute()
            
            current_value = config[setting]
            data_type = inquirer.select(
                message=f"Data type for '{setting}':",
                choices=["String", "Number", "Boolean"],
                default="String"
            ).execute()
            
            if data_type == "String":
                new_value = inquirer.text(
                    message=f"New value for '{setting}':",
                    default=str(current_value)
                ).execute()
            elif data_type == "Number":
                new_value = inquirer.number(
                    message=f"New value for '{setting}':",
                    default=float(current_value) if isinstance(current_value, (int, float)) else 0
                ).execute()
            elif data_type == "Boolean":
                new_value = inquirer.confirm(
                    message=f"Enable '{setting}'?",
                    default=bool(current_value)
                ).execute()
            
            config[setting] = new_value
            print(f"✅ Updated {setting} = {new_value}")
            
        elif action == "add":
            key = inquirer.text(
                message="Setting name:",
                validate=lambda k: len(k) > 0 and k not in config,
                invalid_message="Setting name must be unique and non-empty"
            ).execute()
            
            data_type = inquirer.select(
                message="Data type:",
                choices=["String", "Number", "Boolean"]
            ).execute()
            
            if data_type == "String":
                value = inquirer.text(message="Value:").execute()
            elif data_type == "Number":
                value = inquirer.number(message="Value:").execute()
            elif data_type == "Boolean":
                value = inquirer.confirm(message="Enable?").execute()
            
            config[key] = value
            print(f"✅ Added {key} = {value}")
            
        elif action == "delete":
            if not config:
                print("⚠️  No settings to delete")
                continue
                
            setting = inquirer.select(
                message="Select setting to delete:",
                choices=list(config.keys())
            ).execute()
            
            confirm = inquirer.confirm(
                message=f"Delete '{setting}'?",
                default=False
            ).execute()
            
            if confirm:
                del config[setting]
                print(f"✅ Deleted {setting}")
                
        elif action == "save":
            with open(config_file, 'w') as f:
                json.dump(config, f, indent=2)
            print(f"✅ Configuration saved to {config_file}")
            break
            
        elif action == "exit":
            save_changes = inquirer.confirm(
                message="Save changes before exiting?",
                default=True
            ).execute()
            
            if save_changes:
                with open(config_file, 'w') as f:
                    json.dump(config, f, indent=2)
                print(f"✅ Configuration saved to {config_file}")
            
            break

# Run configuration manager
manage_config()
```

## Testing Interactive Prompts

### Unit Testing with Mock
```python
import unittest
from unittest.mock import patch
from InquirerPy import prompt

def get_user_info():
    questions = [
        {"type": "input", "message": "Name:", "name": "name"},
        {"type": "confirm", "message": "Subscribe?", "name": "subscribe"}
    ]
    return prompt(questions)

class TestPrompts(unittest.TestCase):
    @patch('your_module.prompt')
    def test_get_user_info(self, mock_prompt):
        # Mock the prompt response
        mock_prompt.return_value = {"name": "John", "subscribe": True}
        
        result = get_user_info()
        
        self.assertEqual(result["name"], "John")
        self.assertTrue(result["subscribe"])
        mock_prompt.assert_called_once()

if __name__ == '__main__':
    unittest.main()
```

## Performance and Best Practices

### 1. Choose the Right Library
- **python-inquirer**: Simple, stable, fewer dependencies
- **InquirerPy**: Modern, feature-rich, better styling, async support

### 2. Validation Best Practices
```python
# Good: Clear, specific error messages
def validate_email(email):
    if not email:
        return "Email is required"
    if '@' not in email:
        return "Email must contain @ symbol"
    if not email.endswith(('.com', '.org', '.net')):
        return "Email must end with .com, .org, or .net"
    return True

# Good: Use built-in validators when possible
from InquirerPy.validator import EmptyInputValidator, NumberValidator, PathValidator

# Good: Combine validation with filtering
age = inquirer.text(
    message="Age:",
    validate=NumberValidator(float_allowed=False),
    filter=lambda x: int(x)  # Convert to integer
).execute()
```

### 3. User Experience Tips
```python
# Use clear, action-oriented messages
message="Select deployment environment:"  # Good
message="Environment?"                     # Poor

# Provide helpful instructions
instruction="Use arrow keys to navigate, Enter to select"

# Set sensible defaults
default="production" if is_prod_deploy else "development"

# Use separators to group related options
choices=[
    "Development servers",
    Separator(),
    "dev-01", "dev-02", "dev-03",
    Separator(),
    "Production servers",
    Separator(),
    "prod-01", "prod-02"
]

# Transform output for better UX
transformer=lambda result: f"{len(result)} items selected"
```

### 4. Error Handling
```python
from InquirerPy import inquirer
from InquirerPy.exceptions import InvalidArgument

try:
    result = inquirer.select(
        message="Select option:",
        choices=["A", "B", "C"]
    ).execute()
except KeyboardInterrupt:
    print("\n❌ Operation cancelled by user")
    exit(1)
except InvalidArgument as e:
    print(f"❌ Configuration error: {e}")
    exit(1)
except Exception as e:
    print(f"❌ Unexpected error: {e}")
    exit(1)
```

### 5. Async Support (InquirerPy)
```python
import asyncio
from InquirerPy import prompt_async

async def async_prompts():
    questions = [
        {"type": "input", "message": "Name:"},
        {"type": "confirm", "message": "Continue?"}
    ]
    result = await prompt_async(questions)
    return result

# Run async prompts
result = asyncio.run(async_prompts())
```

This cheat sheet covers both libraries comprehensively, providing you with the tools to create engaging, interactive command-line interfaces. Start with simple prompts and gradually incorporate advanced features like validation, styling, and dynamic behavior as your applications grow in complexity.