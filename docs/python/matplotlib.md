# Matplotlib Cheat Sheet

## Installation

```bash
# Basic installation
pip install matplotlib

# With optional dependencies
pip install matplotlib[complete]

# Check version
python -c "import matplotlib; print(matplotlib.__version__)"

# Check backend
python -c "import matplotlib; print(matplotlib.get_backend())"
```

## Basic Setup

```python
# Essential imports
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np

# Set backend (optional)
matplotlib.use('TkAgg')  # or 'Qt5Agg', 'Agg', etc.

# Enable inline plots in Jupyter
%matplotlib inline

# Basic figure creation
fig, ax = plt.subplots()
plt.show()
```

## Core Functionality

### Basic Plotting

```python
# Line plot
x = np.linspace(0, 10, 100)
y = np.sin(x)

plt.figure(figsize=(8, 6))
plt.plot(x, y, label='sin(x)', linewidth=2, color='blue')
plt.xlabel('X-axis')
plt.ylabel('Y-axis')
plt.title('Simple Sine Wave')
plt.legend()
plt.grid(True)
plt.show()

# Multiple lines
plt.plot(x, np.sin(x), label='sin(x)')
plt.plot(x, np.cos(x), label='cos(x)', linestyle='--')
plt.legend()
```

### Scatter Plot

```python
# Basic scatter
x = np.random.randn(100)
y = np.random.randn(100)
colors = np.random.rand(100)
sizes = 1000 * np.random.rand(100)

plt.scatter(x, y, c=colors, s=sizes, alpha=0.6, cmap='viridis')
plt.colorbar()
plt.title('Scatter Plot with Color and Size')
```

### Bar Charts

```python
# Vertical bar chart
categories = ['A', 'B', 'C', 'D']
values = [23, 17, 35, 29]

plt.bar(categories, values, color=['red', 'green', 'blue', 'orange'])
plt.title('Bar Chart')
plt.ylabel('Values')

# Horizontal bar chart
plt.barh(categories, values)

# Grouped bar chart
x = np.arange(len(categories))
width = 0.35
plt.bar(x - width/2, values, width, label='Series 1')
plt.bar(x + width/2, [20, 25, 15, 30], width, label='Series 2')
plt.xticks(x, categories)
plt.legend()
```

### Histograms

```python
# Basic histogram
data = np.random.normal(100, 15, 1000)
plt.hist(data, bins=30, alpha=0.7, color='skyblue', edgecolor='black')
plt.title('Histogram')
plt.xlabel('Values')
plt.ylabel('Frequency')

# Multiple histograms
data1 = np.random.normal(100, 15, 1000)
data2 = np.random.normal(80, 20, 1000)
plt.hist([data1, data2], bins=30, alpha=0.7, label=['Dataset 1', 'Dataset 2'])
plt.legend()
```

## Subplots and Figure Management

### Creating Subplots

```python
# Basic subplots
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(10, 8))

ax1.plot([1, 2, 3], [1, 4, 2])
ax1.set_title('Plot 1')

ax2.scatter([1, 2, 3], [1, 4, 2])
ax2.set_title('Plot 2')

ax3.bar([1, 2, 3], [1, 4, 2])
ax3.set_title('Plot 3')

ax4.hist(np.random.randn(100), bins=20)
ax4.set_title('Plot 4')

plt.tight_layout()  # Adjust spacing
plt.show()

# Subplot with different sizes
fig = plt.figure(figsize=(12, 8))
ax1 = plt.subplot(2, 2, 1)
ax2 = plt.subplot(2, 2, 2)
ax3 = plt.subplot(2, 1, 2)  # Spans two columns
```

### GridSpec for Advanced Layouts

```python
import matplotlib.gridspec as gridspec

fig = plt.figure(figsize=(10, 8))
gs = gridspec.GridSpec(3, 3)

ax1 = fig.add_subplot(gs[0, :])    # Top row, all columns
ax2 = fig.add_subplot(gs[1, :-1])  # Middle row, first two columns
ax3 = fig.add_subplot(gs[1:, -1])  # Right column, bottom two rows
ax4 = fig.add_subplot(gs[-1, 0])   # Bottom left
ax5 = fig.add_subplot(gs[-1, -2])  # Bottom center
```

### Subplot with Shared Axes

```python
# Shared x-axis
fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(8, 6))
ax1.plot(x, np.sin(x))
ax2.plot(x, np.cos(x))

# Shared y-axis
fig, (ax1, ax2) = plt.subplots(1, 2, sharey=True, figsize=(10, 4))
```

## Customization

### Colors and Styles

```python
# Color specifications
plt.plot(x, y, color='red')           # Named color
plt.plot(x, y, color='#FF5733')       # Hex code
plt.plot(x, y, color=(0.1, 0.2, 0.5)) # RGB tuple
plt.plot(x, y, c='r')                 # Short form

# Line styles
plt.plot(x, y, linestyle='-')         # Solid
plt.plot(x, y, linestyle='--')        # Dashed
plt.plot(x, y, linestyle='-.')        # Dash-dot
plt.plot(x, y, linestyle=':')         # Dotted
plt.plot(x, y, ls=':')                # Short form

# Markers
plt.plot(x, y, marker='o')            # Circle
plt.plot(x, y, marker='s')            # Square
plt.plot(x, y, marker='^')            # Triangle up
plt.plot(x, y, marker='D')            # Diamond

# Combined format string
plt.plot(x, y, 'ro-')  # Red circles with solid line
plt.plot(x, y, 'g--^') # Green dashed line with triangle markers
```

### Fonts and Text

```python
# Font properties
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.size'] = 12

# Title and labels with custom fonts
plt.title('Title', fontsize=16, fontweight='bold')
plt.xlabel('X Label', fontsize=14, style='italic')
plt.ylabel('Y Label', fontsize=14, color='red')

# Text annotations
plt.text(0.5, 0.5, 'Sample Text', fontsize=12, 
         horizontalalignment='center', verticalalignment='center',
         transform=ax.transAxes)  # Relative coordinates

# Annotations with arrows
plt.annotate('Important Point', xy=(2, 1), xytext=(3, 4),
             arrowprops=dict(arrowstyle='->', color='red', lw=2))
```

### Axis Customization

```python
# Axis limits
plt.xlim(0, 10)
plt.ylim(-1, 1)
# or
ax.set_xlim(0, 10)
ax.set_ylim(-1, 1)

# Axis ticks
plt.xticks([0, 2, 4, 6, 8, 10])
plt.yticks([-1, -0.5, 0, 0.5, 1])

# Custom tick labels
plt.xticks([0, 1, 2, 3], ['A', 'B', 'C', 'D'])

# Tick formatting
from matplotlib.ticker import FuncFormatter
def currency(x, pos):
    return f'${x:.0f}'
ax.yaxis.set_major_formatter(FuncFormatter(currency))

# Logarithmic scale
plt.yscale('log')
plt.xscale('log')

# Grid customization
plt.grid(True, linestyle='--', alpha=0.7, color='gray')
```

### Legends

```python
# Basic legend
plt.plot(x, np.sin(x), label='sin(x)')
plt.plot(x, np.cos(x), label='cos(x)')
plt.legend()

# Legend positioning
plt.legend(loc='upper right')  # 'upper left', 'lower right', etc.
plt.legend(loc='best')         # Automatic best position
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')  # Outside plot

# Legend customization
plt.legend(frameon=True, fancybox=True, shadow=True, 
           ncol=2, fontsize=10, title='Legend Title')
```

## Different Plot Types

### Statistical Plots

```python
# Box plot
data = [np.random.normal(0, std, 100) for std in range(1, 4)]
plt.boxplot(data, labels=['Group A', 'Group B', 'Group C'])

# Violin plot (requires seaborn or custom implementation)
# Error bars
x = [1, 2, 3, 4]
y = [1, 4, 2, 3]
yerr = [0.1, 0.2, 0.1, 0.3]
plt.errorbar(x, y, yerr=yerr, fmt='o-', capsize=5)

# Fill between
x = np.linspace(0, 10, 100)
y1 = np.sin(x)
y2 = np.cos(x)
plt.fill_between(x, y1, y2, alpha=0.3, color='blue')
```

### 2D Plots

```python
# Heatmap/Image
data = np.random.rand(10, 10)
plt.imshow(data, cmap='hot', interpolation='nearest')
plt.colorbar()

# Contour plot
x = np.linspace(-3, 3, 100)
y = np.linspace(-3, 3, 100)
X, Y = np.meshgrid(x, y)
Z = np.sin(X) * np.cos(Y)

plt.contour(X, Y, Z, levels=10)
plt.contourf(X, Y, Z, levels=20, cmap='viridis')  # Filled contours
plt.colorbar()

# 3D plotting
from mpl_toolkits.mplot3d import Axes3D
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(X, Y, Z, cmap='viridis')
```

### Pie Charts

```python
# Basic pie chart
sizes = [25, 30, 15, 30]
labels = ['A', 'B', 'C', 'D']
colors = ['gold', 'yellowgreen', 'lightcoral', 'lightskyblue']
explode = (0, 0.1, 0, 0)  # Explode slice B

plt.pie(sizes, explode=explode, labels=labels, colors=colors,
        autopct='%1.1f%%', shadow=True, startangle=90)
plt.axis('equal')  # Equal aspect ratio
```

## Saving Figures

```python
# Save in different formats
plt.savefig('plot.png', dpi=300, bbox_inches='tight')
plt.savefig('plot.pdf', format='pdf')
plt.savefig('plot.svg', format='svg')
plt.savefig('plot.eps', format='eps')

# High-quality publication figure
plt.savefig('publication.png', dpi=300, bbox_inches='tight', 
            facecolor='white', edgecolor='none', transparent=False)

# Save with specific size
plt.figure(figsize=(10, 6))
# ... plotting code ...
plt.savefig('sized_plot.png', dpi=150)
```

## Interactive Features

### Event Handling

```python
# Click event handling
def onclick(event):
    if event.inaxes is not None:
        print(f'Clicked at: ({event.xdata:.2f}, {event.ydata:.2f})')

fig, ax = plt.subplots()
ax.plot([1, 2, 3], [1, 4, 2])
fig.canvas.mpl_connect('button_press_event', onclick)
plt.show()

# Key press events
def onkey(event):
    print(f'Key pressed: {event.key}')

fig.canvas.mpl_connect('key_press_event', onkey)
```

### Widgets (requires widget backend)

```python
from matplotlib.widgets import Button, Slider

# Interactive slider
fig, ax = plt.subplots()
plt.subplots_adjust(bottom=0.25)

# Initial plot
t = np.linspace(0, 10, 1000)
a0 = 1
f0 = 1
s = a0 * np.sin(2 * np.pi * f0 * t)
l, = plt.plot(t, s)

# Slider
ax_freq = plt.axes([0.25, 0.1, 0.5, 0.03])
slider = Slider(ax_freq, 'Frequency', 0.1, 5.0, valinit=f0)

def update(val):
    freq = slider.val
    l.set_ydata(a0 * np.sin(2 * np.pi * freq * t))
    fig.canvas.draw_idle()

slider.on_changed(update)
plt.show()
```

## Advanced Features

### Animations

```python
from matplotlib.animation import FuncAnimation

# Animated sine wave
fig, ax = plt.subplots()
x = np.linspace(0, 2*np.pi, 100)
line, = ax.plot(x, np.sin(x))

def animate(frame):
    line.set_ydata(np.sin(x + frame/10))
    return line,

ani = FuncAnimation(fig, animate, frames=200, interval=50, blit=True)
plt.show()

# Save animation
ani.save('animation.gif', writer='pillow', fps=20)
ani.save('animation.mp4', writer='ffmpeg', fps=30)
```

### Custom Colormaps

```python
from matplotlib.colors import LinearSegmentedColormap

# Create custom colormap
colors = ['red', 'yellow', 'green', 'blue']
n_bins = 100
cmap = LinearSegmentedColormap.from_list('custom', colors, N=n_bins)

# Use custom colormap
data = np.random.rand(10, 10)
plt.imshow(data, cmap=cmap)
plt.colorbar()
```

### Dual Axes

```python
# Two different y-axes
fig, ax1 = plt.subplots()

x = np.linspace(0, 10, 100)
y1 = np.sin(x)
y2 = np.exp(x/5)

# First y-axis
ax1.plot(x, y1, 'g-')
ax1.set_xlabel('X data')
ax1.set_ylabel('sin(x)', color='g')
ax1.tick_params(axis='y', labelcolor='g')

# Second y-axis
ax2 = ax1.twinx()
ax2.plot(x, y2, 'b-')
ax2.set_ylabel('exp(x/5)', color='b')
ax2.tick_params(axis='y', labelcolor='b')

plt.show()
```

## Customization & Best Practices

### Style Sheets

```python
# Available styles
print(plt.style.available)

# Use built-in styles
plt.style.use('seaborn-v0_8')
plt.style.use('ggplot')
plt.style.use('dark_background')

# Use multiple styles
plt.style.use(['seaborn-v0_8', 'seaborn-v0_8-darkgrid'])

# Context manager for temporary style
with plt.style.context('bmh'):
    plt.plot([1, 2, 3], [1, 4, 2])
    plt.show()
```

### RC Parameters

```python
# Global settings
plt.rcParams['figure.figsize'] = [10, 6]
plt.rcParams['font.size'] = 12
plt.rcParams['lines.linewidth'] = 2
plt.rcParams['grid.alpha'] = 0.3

# Context manager for temporary settings
with plt.rc_context({'font.size': 14, 'lines.linewidth': 3}):
    plt.plot([1, 2, 3], [1, 4, 2])

# Reset to defaults
plt.rcdefaults()
```

### Performance Tips

```python
# Use appropriate backends
import matplotlib
matplotlib.use('Agg')  # For server/batch processing

# Batch plotting
plt.ioff()  # Turn off interactive mode
for i in range(100):
    plt.figure()
    plt.plot(data[i])
    plt.savefig(f'plot_{i}.png')
    plt.close()  # Important: close figures to free memory

# Efficient line collection for many lines
from matplotlib.collections import LineCollection
lines = [np.column_stack([x, y + i]) for i in range(100)]
lc = LineCollection(lines, linewidths=0.5)
ax.add_collection(lc)
```

## Integration with Other Libraries

### NumPy Integration

```python
# Matplotlib works seamlessly with NumPy arrays
x = np.linspace(0, 10, 100)
y = np.sin(x)
plt.plot(x, y)

# Masked arrays
y_masked = np.ma.masked_where(y < 0, y)
plt.plot(x, y_masked)
```

### Pandas Integration

```python
import pandas as pd

# DataFrame plotting
df = pd.DataFrame({
    'x': range(10),
    'y1': np.random.randn(10),
    'y2': np.random.randn(10)
})

# Direct pandas plotting
df.plot(x='x', y=['y1', 'y2'])

# Using matplotlib directly
plt.plot(df['x'], df['y1'], label='y1')
plt.plot(df['x'], df['y2'], label='y2')
plt.legend()
```

## Common Gotchas & Best Practices

### Memory Management

```python
# Always close figures when done
fig, ax = plt.subplots()
# ... plotting code ...
plt.close(fig)  # or plt.close('all')

# Clear current figure
plt.clf()

# Clear current axes
plt.cla()
```

### Backend Issues

```python
# Check current backend
print(plt.get_backend())

# Set backend before importing pyplot
import matplotlib
matplotlib.use('TkAgg')  # Must be before importing pyplot
import matplotlib.pyplot as plt
```

### Common Patterns

```python
# Professional plotting setup
def setup_plot(figsize=(10, 6)):
    fig, ax = plt.subplots(figsize=figsize)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(True, alpha=0.3)
    return fig, ax

# Context manager for consistent styling
from contextlib import contextmanager

@contextmanager
def plot_style(style='seaborn-v0_8', figsize=(10, 6)):
    with plt.style.context(style):
        fig, ax = plt.subplots(figsize=figsize)
        yield fig, ax
        plt.tight_layout()

# Usage
with plot_style() as (fig, ax):
    ax.plot([1, 2, 3], [1, 4, 2])
    ax.set_title('Styled Plot')
    plt.show()
```

## Quick Reference

### Essential Functions

| Function | Purpose | Example |
|----------|---------|---------|
| `plt.figure()` | Create new figure | `plt.figure(figsize=(8,6))` |
| `plt.subplot()` | Add subplot | `plt.subplot(2,2,1)` |
| `plt.plot()` | Line plot | `plt.plot(x, y, 'r-')` |
| `plt.scatter()` | Scatter plot | `plt.scatter(x, y, c=colors)` |
| `plt.bar()` | Bar chart | `plt.bar(x, height)` |
| `plt.hist()` | Histogram | `plt.hist(data, bins=20)` |
| `plt.imshow()` | Display image/2D data | `plt.imshow(array, cmap='hot')` |
| `plt.savefig()` | Save figure | `plt.savefig('plot.png', dpi=300)` |

### Format Strings

| Code | Meaning | Code | Meaning |
|------|---------|------|---------|
| `-` | Solid line | `o` | Circle marker |
| `--` | Dashed line | `s` | Square marker |
| `-.` | Dash-dot line | `^` | Triangle up marker |
| `:` | Dotted line | `D` | Diamond marker |
| `r` | Red | `g` | Green |
| `b` | Blue | `k` | Black |
| `c` | Cyan | `m` | Magenta |
| `y` | Yellow | `w` | White |

### Color Maps

Common colormaps: `viridis`, `plasma`, `inferno`, `magma`, `coolwarm`, `RdYlBu`, `seismic`, `hot`, `cool`, `spring`, `summer`, `autumn`, `winter`, `gray`

Use: `plt.imshow(data, cmap='viridis')`