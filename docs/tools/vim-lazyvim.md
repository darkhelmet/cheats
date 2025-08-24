# Vim/Neovim with LazyVim Cheat Sheet

A comprehensive reference for Vim/Neovim fundamentals and LazyVim-specific features. This cheat sheet covers essential motions, commands, and LazyVim's modern enhancements for efficient text editing.

## Quick Start

### LazyVim Installation
```bash
# Backup existing config
mv ~/.config/nvim ~/.config/nvim.bak
mv ~/.local/share/nvim ~/.local/share/nvim.bak

# Clone LazyVim starter
git clone https://github.com/LazyVim/starter ~/.config/nvim
rm -rf ~/.config/nvim/.git

# Start Neovim (plugins install automatically)
nvim
```

### Essential LazyVim Settings
```lua
-- Disable animations (add to lua/config/options.lua)
vim.g.snacks_animate = false

-- Disable auto-formatting
vim.g.autoformat = false

-- Disable lazygit theme sync
vim.g.lazygit_config = false

-- Use basedpyright for Python (development version)
vim.g.lazyvim_python_lsp = "basedpyright"
```

## Core Vim Concepts

### Modes
- **Normal Mode**: Navigation and text manipulation (default)
- **Insert Mode**: Text insertion (`i`, `I`, `a`, `A`, `o`, `O`)
- **Visual Mode**: Text selection (`v`, `V`, `Ctrl-v`)
- **Command Mode**: Ex commands (`:`)

### Mode Transitions
```
Normal → Insert: i, I, a, A, o, O, s, S, c, C
Insert → Normal: <Esc>, Ctrl-c
Normal → Visual: v, V, Ctrl-v
Normal → Command: :, /, ?
Any Mode → Normal: <Esc>
```

## Navigation & Motions

### Basic Movement
| Key | Action |
|-----|--------|
| `h` | Left |
| `j` | Down |
| `k` | Up |
| `l` | Right |
| `w` | Next word start |
| `W` | Next WORD start (space-delimited) |
| `e` | Next word end |
| `E` | Next WORD end |
| `b` | Previous word start |
| `B` | Previous WORD start |
| `ge` | Previous word end |

### Line Navigation
| Key | Action |
|-----|--------|
| `0` | Beginning of line |
| `^` | First non-blank character |
| `$` | End of line |
| `g_` | Last non-blank character |
| `gj` | Down by screen line (wrapped) |
| `gk` | Up by screen line (wrapped) |
| `g0` | Beginning of screen line |
| `g$` | End of screen line |

### File Navigation
| Key | Action |
|-----|--------|
| `gg` | Go to first line |
| `G` | Go to last line |
| `{number}G` | Go to line number |
| `:{number}` | Go to line number |
| `H` | Top of screen |
| `M` | Middle of screen |
| `L` | Bottom of screen |
| `Ctrl-u` | Scroll up half screen |
| `Ctrl-d` | Scroll down half screen |
| `Ctrl-b` | Scroll up full screen |
| `Ctrl-f` | Scroll down full screen |

### Advanced Movement
| Key | Action |
|-----|--------|
| `f{char}` | Find next character in line |
| `F{char}` | Find previous character in line |
| `t{char}` | Till next character |
| `T{char}` | Till previous character |
| `;` | Repeat last f/F/t/T |
| `,` | Repeat last f/F/t/T backward |
| `*` | Search word under cursor forward |
| `#` | Search word under cursor backward |
| `%` | Match parentheses/brackets |
| `(` | Previous sentence |
| `)` | Next sentence |
| `{` | Previous paragraph |
| `}` | Next paragraph |

## Text Objects & Selection

### Text Objects
| Key | Action |
|-----|--------|
| `iw` | Inner word |
| `aw` | A word (with space) |
| `is` | Inner sentence |
| `as` | A sentence |
| `ip` | Inner paragraph |
| `ap` | A paragraph |
| `i"` | Inside quotes |
| `a"` | Around quotes |
| `i'` | Inside single quotes |
| `a'` | Around single quotes |
| `i(` | Inside parentheses |
| `a(` | Around parentheses |
| `i[` | Inside brackets |
| `a[` | Around brackets |
| `i{` | Inside braces |
| `a{` | Around braces |
| `it` | Inside tag |
| `at` | Around tag |

### Visual Mode
| Key | Action |
|-----|--------|
| `v` | Character-wise visual |
| `V` | Line-wise visual |
| `Ctrl-v` | Block-wise visual |
| `o` | Go to other end of selection |
| `gv` | Reselect last visual area |

## Editing Operations

### Insert Mode Entry
| Key | Action |
|-----|--------|
| `i` | Insert before cursor |
| `I` | Insert at beginning of line |
| `a` | Insert after cursor |
| `A` | Insert at end of line |
| `o` | Open line below |
| `O` | Open line above |
| `s` | Substitute character |
| `S` | Substitute line |
| `c{motion}` | Change text |
| `C` | Change to end of line |

### Delete Operations
| Key | Action |
|-----|--------|
| `x` | Delete character under cursor |
| `X` | Delete character before cursor |
| `d{motion}` | Delete text |
| `dd` | Delete line |
| `D` | Delete to end of line |
| `dw` | Delete word |
| `diw` | Delete inner word |
| `d$` | Delete to end of line |
| `d0` | Delete to beginning of line |

### Copy & Paste
| Key | Action |
|-----|--------|
| `y{motion}` | Yank (copy) text |
| `yy` | Yank line |
| `Y` | Yank to end of line |
| `p` | Put (paste) after cursor |
| `P` | Put before cursor |
| `"{register}y` | Yank to register |
| `"{register}p` | Put from register |
| `"0p` | Put from yank register |

### Undo & Redo
| Key | Action |
|-----|--------|
| `u` | Undo |
| `Ctrl-r` | Redo |
| `U` | Undo all changes on line |
| `.` | Repeat last change |

## Search & Replace

### Search
| Key | Action |
|-----|--------|
| `/pattern` | Search forward |
| `?pattern` | Search backward |
| `n` | Next match |
| `N` | Previous match |
| `*` | Search word under cursor forward |
| `#` | Search word under cursor backward |
| `g*` | Search partial word forward |
| `g#` | Search partial word backward |

### Replace
```vim
:s/old/new/         " Replace first in line
:s/old/new/g        " Replace all in line
:%s/old/new/g       " Replace all in file
:%s/old/new/gc      " Replace all with confirmation
:10,20s/old/new/g   " Replace in lines 10-20
```

## LazyVim Key Mappings

### Leader Key
LazyVim uses `<Space>` as the leader key. Press `<Space>` to see all available mappings.

### File Operations
| Key | Action |
|-----|--------|
| `<leader>ff` | Find files |
| `<leader>fg` | Live grep |
| `<leader>fb` | Find buffers |
| `<leader>fh` | Find help |
| `<leader>fr` | Recent files |
| `<leader>fn` | New file |
| `<leader>fe` | File explorer |

### Buffer Management
| Key | Action |
|-----|--------|
| `<S-h>` | Previous buffer |
| `<S-l>` | Next buffer |
| `[b` | Previous buffer |
| `]b` | Next buffer |
| `<leader>bd` | Delete buffer |
| `<leader>bD` | Delete buffer (force) |
| `<leader>bb` | Switch to other buffer |

### Window Management
| Key | Action |
|-----|--------|
| `<Ctrl-w>h` | Move to left window |
| `<Ctrl-w>j` | Move to bottom window |
| `<Ctrl-w>k` | Move to top window |
| `<Ctrl-w>l` | Move to right window |
| `<Ctrl-w>s` | Split horizontally |
| `<Ctrl-w>v` | Split vertically |
| `<Ctrl-w>q` | Close window |
| `<Ctrl-w>=` | Equalize window sizes |
| `<Ctrl-w>_` | Maximize height |
| `<Ctrl-w>|` | Maximize width |

### Code Navigation
| Key | Action |
|-----|--------|
| `gd` | Go to definition |
| `gD` | Go to declaration |
| `gi` | Go to implementation |
| `gy` | Go to type definition |
| `gr` | Go to references |
| `K` | Hover documentation |
| `<leader>ca` | Code actions |
| `<leader>rn` | Rename symbol |
| `[d` | Previous diagnostic |
| `]d` | Next diagnostic |
| `<leader>cd` | Line diagnostics |

### Git Integration
| Key | Action |
|-----|--------|
| `<leader>gg` | LazyGit |
| `<leader>gb` | Git blame |
| `<leader>gf` | Git file history |
| `<leader>gs` | Git status |
| `]h` | Next hunk |
| `[h` | Previous hunk |
| `<leader>ghs` | Stage hunk |
| `<leader>ghu` | Undo hunk |
| `<leader>ghp` | Preview hunk |

### Terminal
| Key | Action |
|-----|--------|
| `<Ctrl-\>` | Toggle terminal |
| `<leader>ft` | Terminal (root dir) |
| `<leader>fT` | Terminal (cwd) |
| `<Ctrl-/>` | Terminal (root dir) |

### Search & Navigation
| Key | Action |
|-----|--------|
| `<leader>/` | Grep (root dir) |
| `<leader>:` | Command history |
| `<leader><space>` | Find files (root dir) |
| `<leader>,` | Switch buffer |
| `<leader>fb` | Buffers |
| `<leader>fc` | Find config file |
| `<leader>ff` | Find files (root dir) |
| `<leader>fF` | Find files (cwd) |
| `<leader>fr` | Recent |
| `<leader>fR` | Recent (cwd) |

### LazyVim Specific
| Key | Action |
|-----|--------|
| `<leader>l` | Lazy (plugin manager) |
| `<leader>L` | LazyVim changelog |
| `<leader>x` | LazyVim extras |

## Advanced Features

### Marks
| Key | Action |
|-----|--------|
| `m{a-z}` | Set local mark |
| `m{A-Z}` | Set global mark |
| `'{mark}` | Jump to mark line |
| `` `{mark}` `` | Jump to mark position |
| `''` | Jump to previous line |
| ``` `` ``` | Jump to previous position |
| `:marks` | List marks |

### Registers
| Key | Action |
|-----|--------|
| `"{a-z}` | Named registers |
| `"0` | Yank register |
| `"1-9` | Delete registers |
| `"+` | System clipboard |
| `"*` | Primary selection |
| `"%` | Current filename |
| `":` | Last command |
| `"/` | Last search |
| `".` | Last inserted text |

### Macros
| Key | Action |
|-----|--------|
| `q{a-z}` | Record macro |
| `q` | Stop recording |
| `@{a-z}` | Play macro |
| `@@` | Repeat last macro |
| `{number}@{a-z}` | Run macro n times |

### Folding
| Key | Action |
|-----|--------|
| `zf{motion}` | Create fold |
| `zd` | Delete fold |
| `zo` | Open fold |
| `zc` | Close fold |
| `za` | Toggle fold |
| `zR` | Open all folds |
| `zM` | Close all folds |
| `zj` | Move to next fold |
| `zk` | Move to previous fold |

## Command Line & Ex Commands

### File Operations
```vim
:w              " Save
:w filename     " Save as
:q              " Quit
:q!             " Quit without saving
:wq or :x       " Save and quit
:e filename     " Edit file
:e!             " Reload file
:enew           " New buffer
```

### Buffer Management
```vim
:ls             " List buffers
:b {number}     " Switch to buffer
:bn             " Next buffer
:bp             " Previous buffer
:bd             " Delete buffer
:bufdo cmd      " Run command on all buffers
```

### Window Management
```vim
:sp filename    " Horizontal split
:vs filename    " Vertical split
:new            " New horizontal split
:vnew           " New vertical split
:only           " Close all other windows
:resize {n}     " Resize window height
:vertical resize {n}  " Resize window width
```

### Advanced Commands
```vim
:set option     " Set option
:set option?    " Show option value
:set option!    " Toggle option
:help topic     " Get help
:history        " Command history
:changes        " Show change list
:jumps          " Show jump list
:reg            " Show registers
```

## Configuration Tips

### Essential LazyVim Options
```lua
-- In lua/config/options.lua
vim.opt.relativenumber = true    -- Relative line numbers
vim.opt.clipboard = "unnamedplus" -- System clipboard
vim.opt.scrolloff = 8            -- Keep 8 lines visible
vim.opt.sidescrolloff = 8        -- Keep 8 columns visible
vim.opt.wrap = false             -- No line wrapping
vim.opt.expandtab = true         -- Use spaces instead of tabs
vim.opt.shiftwidth = 2           -- Indent width
vim.opt.tabstop = 2              -- Tab width
vim.opt.ignorecase = true        -- Ignore case in search
vim.opt.smartcase = true         -- Case sensitive if uppercase used
```

### Custom Keymaps
```lua
-- In lua/config/keymaps.lua
local map = vim.keymap.set

-- Better window navigation
map("n", "<C-h>", "<C-w>h")
map("n", "<C-j>", "<C-w>j")
map("n", "<C-k>", "<C-w>k")
map("n", "<C-l>", "<C-w>l")

-- Stay in indent mode
map("v", "<", "<gv")
map("v", ">", ">gv")

-- Move text up and down
map("v", "J", ":m '>+1<CR>gv=gv")
map("v", "K", ":m '<-2<CR>gv=gv")

-- Better paste
map("v", "p", '"_dP')
```

## Performance Tips

1. **Use relative line numbers** for efficient jumping
2. **Learn text objects** for precise editing
3. **Master the `.` command** for repetition
4. **Use visual mode** for complex selections
5. **Leverage marks** for quick navigation
6. **Practice motions** to avoid arrow keys
7. **Use LazyVim's fuzzy finding** instead of file trees
8. **Learn LSP shortcuts** for code navigation
9. **Use multiple cursors** with visual block mode
10. **Customize based on your workflow**

## Troubleshooting

### Common Issues
```bash
# Check LazyVim health
:checkhealth

# View logs
:messages

# Profile startup time
nvim --startuptime startup.log

# Reset LazyVim
rm -rf ~/.local/share/nvim
rm -rf ~/.local/state/nvim
rm -rf ~/.cache/nvim
```

### Plugin Management
```vim
:Lazy              " Open Lazy.nvim
:Lazy sync         " Update plugins
:Lazy clean        " Remove unused plugins
:Lazy profile      " Profile plugin loading
```

This cheat sheet covers the essential Vim motions and LazyVim enhancements. Practice these commands daily to build muscle memory and improve your editing efficiency. Remember that Vim's power comes from combining simple commands into complex operations.