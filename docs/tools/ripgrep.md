# Ripgrep (rg) Cheat Sheet

**ripgrep** is a line-oriented search tool that recursively searches the current directory for a regex pattern. It's written in Rust and is significantly faster than grep, ag, and other alternatives while respecting `.gitignore` rules by default.

## Quick Start

### Installation
```bash
# macOS (Homebrew)
brew install ripgrep

# Ubuntu/Debian
sudo apt-get install ripgrep

# Fedora
sudo dnf install ripgrep

# Arch Linux
sudo pacman -S ripgrep

# Windows (Chocolatey)
choco install ripgrep

# Windows (Scoop)
scoop install ripgrep

# From source (requires Rust)
cargo install ripgrep
```

### Basic Syntax
```bash
rg <pattern> [path]                    # Search for pattern in current directory
rg "hello world"                       # Search for exact phrase
rg hello README.md                     # Search in specific file
rg hello /path/to/directory           # Search in specific directory
```

## Core Concepts

### Default Behavior
- **Recursive**: Searches all subdirectories by default
- **Gitignore-aware**: Respects `.gitignore`, `.ignore`, and `.rgignore` files
- **Binary file filtering**: Skips binary files automatically
- **Hidden file filtering**: Ignores hidden files and directories
- **Unicode support**: Full Unicode support with proper handling

### Disable Automatic Filtering
```bash
rg -uuu pattern                       # Disable all filtering (gitignore, hidden, binary)
rg -u pattern                         # Disable gitignore filtering
rg -uu pattern                        # Also search hidden files
rg --no-ignore pattern                # Disable gitignore filtering
rg --hidden pattern                   # Search hidden files
rg -a pattern                         # Search binary files as text
```

## Common Search Patterns

### Basic Patterns
```bash
rg "exact phrase"                     # Exact phrase search
rg word                               # Single word search
rg -i case                           # Case-insensitive search
rg -S smart                          # Smart case (case-insensitive unless pattern has uppercase)
rg -w function                       # Word boundary search
rg -F "literal.string"               # Fixed string (no regex)
```

### Regular Expressions
```bash
rg "func.*main"                      # Any characters between func and main
rg "^import"                         # Lines starting with import
rg "error$"                          # Lines ending with error
rg "\d{3}-\d{4}"                     # Phone number pattern (XXX-XXXX)
rg "TODO|FIXME|BUG"                  # Multiple patterns (OR)
rg "\w+@\w+\.\w+"                    # Basic email pattern
```

### Advanced Regex Features
```bash
# Use PCRE2 engine for advanced features
rg -P "(?<=func\s)\w+"               # Positive lookbehind
rg -P "(\w+)\1"                      # Backreferences
rg -P "\b\w+(?=\s+error)"            # Positive lookahead

# Multiline patterns
rg -U "function.*\{.*\}"             # Multiline function pattern
rg -U "struct.*\{[^}]*field"         # Struct with specific field
```

## File Filtering and Type Selection

### File Types
```bash
# List all available file types
rg --type-list

# Search specific file types
rg pattern -tpy                      # Python files only
rg pattern -tjs                      # JavaScript files only
rg pattern -trust                    # Rust files only
rg pattern -tc                       # C files only

# Exclude file types
rg pattern -Trust                    # Exclude Rust files
rg pattern -Tjs                      # Exclude JavaScript files
rg pattern --type-not rust           # Alternative syntax
```

### Glob Patterns
```bash
# Include files matching pattern
rg pattern -g "*.toml"               # TOML files only
rg pattern -g "*.{py,pyx}"           # Python files
rg pattern -g "test_*.py"            # Test files

# Exclude files matching pattern
rg pattern -g "!*.min.js"            # Exclude minified JS
rg pattern -g "!node_modules/*"      # Exclude node_modules
rg pattern -g "!*.log"               # Exclude log files

# Multiple glob patterns
rg pattern -g "*.rs" -g "!target/*"  # Rust files, exclude target directory
```

### Custom File Types
```bash
# Define custom file type
rg pattern --type-add 'web:*.{html,css,js}' -tweb

# Make persistent with alias
alias rg="rg --type-add 'web:*.{html,css,js}'"

# Using configuration file
echo "--type-add=web:*.{html,css,js}" >> ~/.config/ripgrep/rc
export RIPGREP_CONFIG_PATH="$HOME/.config/ripgrep/rc"
```

## Output Formatting and Options

### Basic Output Control
```bash
rg pattern -n                        # Show line numbers (default)
rg pattern -N                        # Don't show line numbers
rg pattern -H                        # Show file names (default when multiple files)
rg pattern --no-filename             # Don't show file names
rg pattern -c                        # Count matching lines only
rg pattern --count-matches           # Count individual matches
```

### Context and Surrounding Lines
```bash
rg pattern -A 3                      # Show 3 lines after match
rg pattern -B 2                      # Show 2 lines before match
rg pattern -C 2                      # Show 2 lines before and after
rg pattern --context 2               # Alternative syntax for -C
```

### Output Modes
```bash
rg pattern -l                        # List files with matches only
rg pattern --files-with-matches      # Alternative syntax
rg pattern --files-without-match     # List files without matches
rg pattern -o                        # Show only matching parts
rg pattern --only-matching           # Alternative syntax
rg pattern -v                        # Show non-matching lines (invert match)
```

### Column and Statistics
```bash
rg pattern --column                  # Show column numbers
rg pattern --stats                   # Show search statistics
rg pattern --debug                   # Show debug information
rg pattern --trace                   # Show trace information
```

## Text Replacement and Substitution

### Basic Replacement
```bash
# Show what replacements would look like (no file modification)
rg "fast" -r "FAST" README.md        # Replace fast with FAST
rg "fast" --replace "FAST"           # Alternative syntax
```

### Capture Groups
```bash
# Using numbered capture groups
rg "fast\s+(\w+)" -r "fast-$1"       # fast word -> fast-word

# Using named capture groups
rg "fast\s+(?P<word>\w+)" -r "fast-$word"
```

### Whole Line Replacement
```bash
rg "^.*error.*$" -r "ERROR LINE"     # Replace entire lines containing error
```

### File Modification (with external tools)
```bash
# GNU sed (Linux)
rg foo -l | xargs sed -i 's/foo/bar/g'

# BSD sed (macOS)
rg foo -l | xargs sed -i '' 's/foo/bar/g'

# Handle filenames with spaces
rg foo -l -0 | xargs -0 sed -i 's/foo/bar/g'
```

## Color and Visual Customization

### Color Control
```bash
rg pattern --color never             # Disable colors
rg pattern --color always            # Force colors
rg pattern --color auto              # Automatic (default)
rg pattern --color ansi              # Use ANSI colors only
```

### Custom Colors
```bash
# Individual color settings
rg pattern --colors 'match:fg:red'
rg pattern --colors 'path:fg:green'
rg pattern --colors 'line:fg:yellow'
rg pattern --colors 'column:fg:blue'

# Multiple color settings
rg pattern \
  --colors 'match:fg:white' \
  --colors 'match:bg:blue' \
  --colors 'match:style:bold'

# RGB colors
rg pattern --colors 'match:fg:255,0,0'   # Bright red
rg pattern --colors 'match:bg:0x33,0x66,0xFF'  # Hex colors

# Clear default styles first (recommended)
rg pattern \
  --colors 'match:none' \
  --colors 'match:fg:blue'
```

### Configuration File for Colors
```bash
# ~/.config/ripgrep/rc
--colors=line:fg:yellow
--colors=line:style:bold
--colors=path:fg:green
--colors=path:style:bold
--colors=match:fg:black
--colors=match:bg:yellow
--colors=match:style:nobold

# Use configuration
export RIPGREP_CONFIG_PATH="$HOME/.config/ripgrep/rc"
```

## Performance Tips and Best Practices

### Speed Optimization
```bash
# Use fixed strings when possible (faster than regex)
rg -F "exact string"                 # Much faster for literal searches

# Limit search scope
rg pattern src/                      # Search specific directory
rg pattern -trs                     # Limit to Rust files

# Use word boundaries for whole words
rg -w function                       # Faster than "\\bfunction\\b"

# Smart case by default
rg -S pattern                        # Case insensitive unless uppercase in pattern
```

### Memory and Resource Control
```bash
# Limit line length to prevent huge output
rg pattern --max-columns 150
rg pattern --max-columns-preview     # Show preview of long lines

# Limit file size to search
rg pattern --max-filesize 1M         # Skip files larger than 1MB

# Control number of threads
rg pattern -j 4                      # Use 4 threads
rg pattern --threads 1               # Single-threaded

# Memory mapping control
rg pattern --mmap                    # Force memory mapping
rg pattern --no-mmap                 # Disable memory mapping
```

### Large File and Binary Handling
```bash
# Search compressed files
rg pattern -z                        # Search gzip, bzip2, xz, lzma, lz4, brotli, zstd

# Handle encoding
rg pattern --encoding utf8           # Force UTF-8
rg pattern --encoding none           # No encoding (binary search)

# Binary file handling
rg pattern -a                        # Search binary files as text
rg pattern --binary                  # Show binary matches
```

## Advanced Features

### Preprocessing
```bash
# Use preprocessor for special file types (e.g., PDFs)
rg pattern --pre ./preprocess.sh

# Conditional preprocessing with glob
rg pattern --pre ./preprocess.sh --pre-glob '*.pdf'
```

### Example preprocessor script:
```bash
#!/bin/sh
# preprocess.sh
case "$1" in
*.pdf)
  if [ -s "$1" ]; then
    exec pdftotext "$1" -
  else
    exec cat
  fi
  ;;
*)
  exec cat
  ;;
esac
```

### Regular Expression Engine Control
```bash
# Default engine (fast, limited features)
rg pattern                           # Default Rust regex engine

# PCRE2 engine (slower, more features)
rg -P pattern                        # Enable lookarounds, backreferences
rg -P --no-pcre2-unicode pattern     # Disable Unicode for speed

# Engine size limits
rg pattern --regex-size-limit 1G     # Increase regex compilation size
rg pattern --dfa-size-limit 1G       # Increase DFA cache size
```

### Multiline Search
```bash
rg -U "function.*{.*return.*}"       # Search across line boundaries
rg -U --multiline-dotall "start.*end" # . matches newlines too
```

## Integration with Other Tools

### Shell Integration
```bash
# Generate shell completions
rg --generate complete-bash > ~/.local/share/bash-completion/completions/rg
rg --generate complete-zsh > ~/.zsh/completions/_rg
rg --generate complete-fish > ~/.config/fish/completions/rg.fish

# Generate man page
rg --generate man | man -l -
```

### Editor Integration
```bash
# Vim/Neovim grepprg
set grepprg=rg\ --vimgrep\ --no-heading\ --smart-case

# VS Code search with ripgrep
"search.useRipgrep": true

# Emacs
(setq counsel-rg-base-command "rg -i -M 120 --no-heading --line-number --color never %s")
```

### Pipeline Usage
```bash
# Find and process files
rg -l "TODO" | head -5              # First 5 files with TODO
rg -l "pattern" | wc -l             # Count files with pattern
rg -c "error" | sort -t: -k2 -nr    # Sort by match count

# Combining with other tools
rg "class.*:" --only-matching | sort | uniq -c  # Count class definitions
rg "import.*from" -o | awk '{print $3}' | sort | uniq  # List import sources
```

## Common Command Combinations

### Development Workflow
```bash
# Find TODOs and FIXMEs in code
rg "(TODO|FIXME|BUG|HACK|XXX)" -n

# Search for functions/methods
rg "(def|function|fn)\s+\w+" -tpy -tjs -trust

# Find unused imports (basic)
rg "^import.*" --only-matching | sort | uniq -c | awk '$1==1'

# Search for potential security issues
rg -i "(password|secret|key|token).*=" -tpy -tjs

# Find large functions (rough estimate)
rg -U "^(def|function|fn).*\{" -A 50 | rg -c "^}" | awk -F: '$2>30'
```

### System Administration
```bash
# Search log files
rg "ERROR|FATAL" /var/log/ -tlog

# Find configuration issues
rg -i "(error|fail|exception)" /etc/ -g "*.conf" -g "*.cfg"

# Process and network searches
rg ":80\b" /etc/ -n                 # Find port 80 references
rg "127\.0\.0\.1" /etc/ -n          # Find localhost references
```

### Data Analysis
```bash
# Count occurrences
rg "pattern" -c | awk -F: '{sum+=$2} END {print sum}'

# Extract structured data
rg "\b\d{4}-\d{2}-\d{2}\b" -o      # Extract dates
rg "\b[\w.-]+@[\w.-]+\.\w+\b" -o   # Extract emails
rg "\b(?:[0-9]{1,3}\.){3}[0-9]{1,3}\b" -o  # Extract IP addresses
```

## Troubleshooting and Gotchas

### Common Issues
```bash
# Pattern not found due to gitignore
rg pattern --no-ignore              # Disable gitignore rules

# Searching binary files
rg pattern -a                       # Force text mode

# Too many matches
rg pattern --max-count 10           # Limit matches per file

# Permission denied errors
rg pattern 2>/dev/null              # Suppress error messages
```

### Performance Issues
```bash
# Slow searches in large repositories
rg pattern --max-filesize 1M        # Skip large files
rg pattern -j 1                     # Use single thread
rg pattern --no-mmap                # Disable memory mapping

# Memory issues with large files
rg pattern --max-columns 100        # Limit line length
rg pattern --multiline-dotall false # Disable multiline optimizations
```

### Platform-Specific Issues
```bash
# Windows path issues (Cygwin)
MSYS_NO_PATHCONV=1 rg "/pattern"

# PowerShell encoding issues
$OutputEncoding = [System.Text.UTF8Encoding]::new()
```

## Configuration Files

### Global Configuration
Create `~/.config/ripgrep/rc`:
```bash
# Default options
--smart-case
--hidden
--max-columns=150
--max-columns-preview

# Custom file types
--type-add=web:*.{html,css,js,jsx,ts,tsx}
--type-add=config:*.{json,yaml,yml,toml,ini}

# Color settings
--colors=line:fg:yellow
--colors=path:fg:green
--colors=match:bg:blue
--colors=match:fg:white

# Exclusions
--glob=!.git/*
--glob=!node_modules/*
--glob=!target/*
```

Set environment variable:
```bash
export RIPGREP_CONFIG_PATH="$HOME/.config/ripgrep/rc"
```

### Project-specific Configuration
Create `.rgignore` in project root:
```bash
# Ignore build artifacts
target/
build/
dist/

# Ignore dependencies
node_modules/
vendor/

# Ignore logs
*.log
logs/
```

## Quick Reference

### Most Useful Flags
| Flag | Purpose |
|------|---------|
| `-i` | Case insensitive |
| `-S` | Smart case |
| `-w` | Word boundaries |
| `-F` | Fixed strings (literal) |
| `-v` | Invert match |
| `-c` | Count matches |
| `-l` | List files with matches |
| `-n` | Line numbers |
| `-A/B/C` | Context lines |
| `-t<type>` | File type filter |
| `-g` | Glob pattern |
| `--hidden` | Search hidden files |
| `--no-ignore` | Ignore .gitignore |
| `-u/-uu/-uuu` | Reduce filtering |

### Essential Patterns
| Pattern | Matches |
|---------|---------|
| `\b\w+\b` | Whole words |
| `^\s*$` | Empty lines |
| `\d+` | Numbers |
| `[A-Z_]+` | Constants |
| `https?://\S+` | URLs |
| `\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z\|a-z]{2,}\b` | Emails |

This cheat sheet covers the most important ripgrep features and usage patterns. For exhaustive documentation, use `rg --help` or generate the man page with `rg --generate man`.