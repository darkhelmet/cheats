# Seaborn

## Installation

```bash
# Basic installation
pip install seaborn

# With all optional dependencies
pip install seaborn[all]

# Development version
pip install git+https://github.com/mwaskom/seaborn.git

# Check version
python -c "import seaborn as sns; print(sns.__version__)"

# List available datasets
python -c "import seaborn as sns; print(sns.get_dataset_names())"
```

## Basic Setup

```python
# Essential imports
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# Apply default theme
sns.set_theme()

# Alternative: set specific style
sns.set_theme(style="whitegrid", palette="pastel")

# Load sample dataset
tips = sns.load_dataset("tips")
flights = sns.load_dataset("flights")
iris = sns.load_dataset("iris")
penguins = sns.load_dataset("penguins")
```

## Core Functionality

### Built-in Datasets

```python
# Available datasets
dataset_names = sns.get_dataset_names()
print(dataset_names)

# Load specific datasets
tips = sns.load_dataset("tips")           # Restaurant tips
flights = sns.load_dataset("flights")     # Airline passenger data
iris = sns.load_dataset("iris")           # Iris flower measurements
penguins = sns.load_dataset("penguins")   # Palmer penguin data
mpg = sns.load_dataset("mpg")             # Car fuel efficiency
titanic = sns.load_dataset("titanic")     # Titanic passenger data
diamonds = sns.load_dataset("diamonds")   # Diamond characteristics
fmri = sns.load_dataset("fmri")           # fMRI brain imaging data

# Explore dataset structure
print(tips.head())
print(tips.info())
print(tips.describe())
```

### Figure-Level vs Axes-Level Functions

```python
# Figure-level functions (create entire figure with subplots)
sns.relplot()     # Relationships (scatter, line)
sns.displot()     # Distributions (hist, kde, ecdf)
sns.catplot()     # Categorical (bar, box, violin, etc.)
sns.lmplot()      # Linear model fits
sns.FacetGrid()   # General-purpose faceting

# Axes-level functions (work with matplotlib axes)
sns.scatterplot() # Scatter plot
sns.lineplot()    # Line plot
sns.histplot()    # Histogram
sns.kdeplot()     # Kernel density estimate
sns.boxplot()     # Box plot
sns.barplot()     # Bar plot
sns.heatmap()     # Heat map
```

## Relational Plots

### Scatter Plots

```python
# Basic scatter plot
sns.scatterplot(data=tips, x="total_bill", y="tip")

# With categorical encoding
sns.scatterplot(data=tips, x="total_bill", y="tip", hue="time", style="smoker")

# With size encoding
sns.scatterplot(data=tips, x="total_bill", y="tip", size="size", hue="time")

# Figure-level with faceting
sns.relplot(data=tips, x="total_bill", y="tip", 
            col="time", hue="smoker", style="smoker")

# Advanced customization
sns.relplot(data=tips, x="total_bill", y="tip",
            hue="time", size="size", style="sex",
            palette=["blue", "red"], sizes=(20, 200),
            height=5, aspect=1.2)
```

### Line Plots

```python
# Basic line plot
fmri = sns.load_dataset("fmri")
sns.lineplot(data=fmri, x="timepoint", y="signal")

# With confidence intervals
sns.lineplot(data=fmri, x="timepoint", y="signal", hue="event")

# Multiple grouping variables
sns.lineplot(data=fmri, x="timepoint", y="signal", 
             hue="region", style="event")

# Figure-level line plots with faceting
sns.relplot(data=fmri, kind="line",
            x="timepoint", y="signal", 
            col="region", hue="event", style="event")

# Time series with dates
flights_wide = flights.pivot(index="year", columns="month", values="passengers")
flights_wide.index = pd.date_range("1949", periods=12, freq="AS")
sns.lineplot(data=flights_wide.T)
```

## Distribution Plots

### Histograms

```python
# Basic histogram
sns.histplot(data=penguins, x="flipper_length_mm")

# With grouping
sns.histplot(data=penguins, x="flipper_length_mm", hue="species")

# Stacked histogram
sns.histplot(data=penguins, x="flipper_length_mm", hue="species", multiple="stack")

# Density histogram
sns.histplot(data=penguins, x="flipper_length_mm", stat="density")

# 2D histogram
sns.histplot(data=penguins, x="flipper_length_mm", y="bill_length_mm")

# Figure-level distributions
sns.displot(data=penguins, x="flipper_length_mm", col="species")
sns.displot(data=penguins, x="flipper_length_mm", hue="species", kind="kde")
```

### KDE Plots

```python
# Basic KDE
sns.kdeplot(data=penguins, x="flipper_length_mm")

# Multiple distributions
sns.kdeplot(data=penguins, x="flipper_length_mm", hue="species")

# Filled KDE
sns.kdeplot(data=penguins, x="flipper_length_mm", hue="species", fill=True)

# 2D KDE
sns.kdeplot(data=penguins, x="flipper_length_mm", y="bill_length_mm")

# Bivariate with contours
sns.kdeplot(data=penguins, x="flipper_length_mm", y="bill_length_mm", 
            levels=5, thresh=0.1)

# Combined histogram and KDE
sns.histplot(data=penguins, x="flipper_length_mm", kde=True)
```

### ECDF Plots

```python
# Empirical Cumulative Distribution Function
sns.ecdfplot(data=penguins, x="flipper_length_mm")

# With grouping
sns.ecdfplot(data=penguins, x="flipper_length_mm", hue="species")

# Complementary ECDF
sns.ecdfplot(data=penguins, x="flipper_length_mm", complementary=True)

# Figure-level ECDF
sns.displot(data=penguins, x="flipper_length_mm", kind="ecdf", 
            col="species", height=4)
```

## Categorical Plots

### Bar Plots

```python
# Basic bar plot (shows mean with confidence interval)
sns.barplot(data=tips, x="day", y="total_bill")

# With grouping
sns.barplot(data=tips, x="day", y="total_bill", hue="time")

# Different estimator
sns.barplot(data=tips, x="day", y="total_bill", estimator=np.median)

# Count plot (frequency of categories)
sns.countplot(data=tips, x="day")
sns.countplot(data=tips, x="day", hue="time")

# Horizontal bar plot
sns.barplot(data=tips, x="total_bill", y="day", orient="h")
```

### Box and Violin Plots

```python
# Box plots
sns.boxplot(data=tips, x="day", y="total_bill")
sns.boxplot(data=tips, x="day", y="total_bill", hue="smoker")

# Violin plots
sns.violinplot(data=tips, x="day", y="total_bill")
sns.violinplot(data=tips, x="day", y="total_bill", hue="smoker", split=True)

# Box plot with strip plot overlay
sns.boxplot(data=tips, x="day", y="total_bill", color="lightgray")
sns.stripplot(data=tips, x="day", y="total_bill", size=4, jitter=True)
```

### Point and Strip Plots

```python
# Strip plot (categorical scatter)
sns.stripplot(data=tips, x="day", y="total_bill")

# Swarm plot (non-overlapping points)
sns.swarmplot(data=tips, x="day", y="total_bill")

# Point plot (connect means)
sns.pointplot(data=tips, x="day", y="total_bill", hue="time")

# Figure-level categorical plots
sns.catplot(data=tips, x="day", y="total_bill", kind="violin", 
            col="time", hue="smoker")

sns.catplot(data=tips, x="day", y="total_bill", kind="swarm", 
            row="time", col="sex")
```

## Statistical Visualizations

### Regression Plots

```python
# Simple linear regression
sns.regplot(data=tips, x="total_bill", y="tip")

# Without regression line
sns.regplot(data=tips, x="total_bill", y="tip", fit_reg=False)

# Different regression order
sns.regplot(data=tips, x="total_bill", y="tip", order=2)

# Logistic regression
sns.regplot(data=tips, x="total_bill", y="tip", logistic=True)

# Linear model plot with faceting
sns.lmplot(data=tips, x="total_bill", y="tip", col="time", hue="smoker")

# Residual plots
sns.residplot(data=tips, x="total_bill", y="tip")
```

### Pair Plots

```python
# Pairwise relationships
sns.pairplot(data=iris)

# With categorical encoding
sns.pairplot(data=iris, hue="species")

# Subset of variables
sns.pairplot(data=iris, vars=["sepal_length", "sepal_width"], hue="species")

# Different plot types on diagonal
sns.pairplot(data=iris, hue="species", diag_kind="kde")

# Custom plot types
sns.pairplot(data=iris, hue="species", 
             plot_kws={"alpha": 0.6}, diag_kws={"shade": True})
```

### Joint Plots

```python
# Basic joint plot
sns.jointplot(data=tips, x="total_bill", y="tip")

# Different plot types
sns.jointplot(data=tips, x="total_bill", y="tip", kind="reg")
sns.jointplot(data=tips, x="total_bill", y="tip", kind="hex")
sns.jointplot(data=tips, x="total_bill", y="tip", kind="kde")

# With categorical data
sns.jointplot(data=penguins, x="flipper_length_mm", y="bill_length_mm", 
              hue="species")

# Custom marginal plots
g = sns.jointplot(data=penguins, x="flipper_length_mm", y="bill_length_mm")
g.plot_joint(sns.kdeplot, color="r", zorder=0, levels=6)
g.plot_marginals(sns.rugplot, color="r", height=-0.15)
```

## Multi-plot Grids

### FacetGrid

```python
# Create FacetGrid
g = sns.FacetGrid(tips, col="time", row="smoker", margin_titles=True)

# Map function to each facet
g.map(sns.scatterplot, "total_bill", "tip", alpha=0.7)
g.add_legend()

# Different functions for different positions
g = sns.FacetGrid(tips, col="time", hue="smoker")
g.map(plt.scatter, "total_bill", "tip", alpha=0.7)
g.add_legend()

# Custom function
def scatter_with_corr(x, y, **kwargs):
    ax = plt.gca()
    corr = np.corrcoef(x, y)[0, 1]
    ax.annotate(f'r = {corr:.2f}', xy=(0.1, 0.9), xycoords=ax.transAxes)
    ax.scatter(x, y, **kwargs)

g = sns.FacetGrid(tips, col="time")
g.map(scatter_with_corr, "total_bill", "tip")
```

### PairGrid

```python
# Create PairGrid
g = sns.PairGrid(iris, hue="species")

# Map different plot types
g.map_diag(sns.histplot)
g.map_offdiag(sns.scatterplot)
g.add_legend()

# Different plots for upper and lower triangles
g = sns.PairGrid(iris)
g.map_upper(sns.scatterplot)
g.map_lower(sns.kdeplot, fill=True)
g.map_diag(sns.histplot, kde=True)
```

### JointGrid

```python
# Create JointGrid
g = sns.JointGrid(data=penguins, x="flipper_length_mm", y="bill_length_mm")

# Add plots
g.plot(sns.scatterplot, sns.histplot)

# Custom styling
g = sns.JointGrid(data=penguins, x="flipper_length_mm", y="bill_length_mm")
g.plot(sns.scatterplot, sns.histplot, alpha=0.7, edgecolor=".2", linewidth=0.5)

# Mixed plot types
g = sns.JointGrid(data=penguins, x="flipper_length_mm", y="bill_length_mm")
g.plot(sns.regplot, sns.boxplot)
```

## Heat Maps

### Basic Heat Maps

```python
# Correlation matrix
corr = tips.corr(numeric_only=True)
sns.heatmap(corr)

# With annotations
sns.heatmap(corr, annot=True, cmap='coolwarm', center=0)

# Custom formatting
sns.heatmap(corr, annot=True, fmt='.2f', square=True, 
            linewidths=0.5, cbar_kws={"shrink": 0.8})

# Pivot table heatmap
flights_pivot = flights.pivot(index="month", columns="year", values="passengers")
sns.heatmap(flights_pivot, cmap="YlOrRd")
```

### Cluster Map

```python
# Hierarchical clustering
iris_num = iris.select_dtypes(include=[np.number])
sns.clustermap(iris_num, cmap='viridis', standard_scale=1)

# With annotations
sns.clustermap(corr, annot=True, cmap='RdBu_r', center=0)

# Control clustering
sns.clustermap(flights_pivot, col_cluster=False, cmap='Blues')
```

## Styling and Themes

### Built-in Themes

```python
# Available styles
print(sns.axes_style.__doc__)

# Set different styles
styles = ['darkgrid', 'whitegrid', 'dark', 'white', 'ticks']

for style in styles:
    sns.set_theme(style=style)
    plt.figure(figsize=(6, 4))
    sns.lineplot(data=fmri.query('region=="frontal"'), 
                 x="timepoint", y="signal", hue="event")
    plt.title(f'Style: {style}')
    plt.show()
```

### Color Palettes

```python
# Qualitative palettes
sns.color_palette("deep")          # Default seaborn colors
sns.color_palette("muted")         # Muted version
sns.color_palette("bright")        # Bright version
sns.color_palette("pastel")        # Pastel version
sns.color_palette("dark")          # Dark version

# Sequential palettes
sns.color_palette("Blues")         # Single hue
sns.color_palette("viridis")       # Perceptually uniform
sns.color_palette("rocket")        # Seaborn sequential

# Diverging palettes
sns.color_palette("RdBu")          # Red-Blue diverging
sns.color_palette("coolwarm")      # Cool-warm
sns.color_palette("vlag")          # Seaborn diverging

# Custom palettes
colors = ["#FF6B6B", "#4ECDC4", "#45B7D1", "#96CEB4", "#FFEAA7"]
sns.set_palette(colors)

# Using palettes in plots
sns.scatterplot(data=tips, x="total_bill", y="tip", hue="time", 
                palette="viridis")
```

### Contexts (Scaling)

```python
# Available contexts
contexts = ['paper', 'notebook', 'talk', 'poster']

for context in contexts:
    sns.set_context(context)
    plt.figure(figsize=(8, 6))
    sns.boxplot(data=tips, x="day", y="total_bill")
    plt.title(f'Context: {context}')
    plt.show()

# Custom scaling
sns.set_context("notebook", font_scale=1.5, rc={"lines.linewidth": 2.5})
```

### Custom Styling

```python
# Custom theme dictionary
custom_theme = {
    "axes.spines.right": False,
    "axes.spines.top": False,
    "axes.grid": True,
    "axes.grid.alpha": 0.3,
    "grid.linewidth": 0.8,
    "font.family": ["serif"],
    "font.size": 12
}

# Apply custom theme
with sns.axes_style(custom_theme):
    sns.scatterplot(data=tips, x="total_bill", y="tip", hue="time")

# Persistent custom theme
sns.set_theme(rc=custom_theme)
```

### Despining

```python
# Remove spines
sns.despine()                      # Remove top and right
sns.despine(left=True)            # Also remove left
sns.despine(offset=10)            # Offset spines
sns.despine(trim=True)            # Trim spines to data range

# In context
plt.figure(figsize=(8, 6))
sns.boxplot(data=tips, x="day", y="total_bill")
sns.despine(offset=5, trim=True)
```

## Advanced Features

### Custom Color Maps and Normalization

```python
# Custom discrete palette
from matplotlib.colors import ListedColormap
custom_colors = ["#FF6B6B", "#4ECDC4", "#45B7D1"]
custom_cmap = ListedColormap(custom_colors)

# Use in heatmap
sns.heatmap(flights_pivot, cmap=custom_cmap)

# Color normalization
from matplotlib.colors import LogNorm, PowerNorm

# Log normalization for highly skewed data
sns.heatmap(flights_pivot, norm=LogNorm())

# Power normalization
sns.heatmap(flights_pivot, norm=PowerNorm(gamma=0.5))
```

### Statistical Annotations

```python
# Add statistical annotations manually
from scipy import stats

fig, ax = plt.subplots()
sns.boxplot(data=tips, x="time", y="total_bill", ax=ax)

# Perform statistical test
lunch_bills = tips[tips['time'] == 'Lunch']['total_bill']
dinner_bills = tips[tips['time'] == 'Dinner']['total_bill']
t_stat, p_value = stats.ttest_ind(lunch_bills, dinner_bills)

# Add annotation
ax.text(0.5, 0.95, f'p-value: {p_value:.4f}', 
        transform=ax.transAxes, ha='center', va='top',
        bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
```

### Interactive Elements

```python
# Using matplotlib widgets with seaborn
from matplotlib.widgets import CheckButtons

# Create plot
fig, ax = plt.subplots(figsize=(10, 6))
species_list = penguins['species'].unique()
lines = []

for species in species_list:
    data = penguins[penguins['species'] == species]
    line = ax.scatter(data['flipper_length_mm'], data['bill_length_mm'], 
                     label=species, alpha=0.7)
    lines.append(line)

# Add checkboxes
rax = plt.axes([0.02, 0.5, 0.15, 0.15])
check = CheckButtons(rax, species_list, [True] * len(species_list))

def toggle_species(label):
    index = species_list.tolist().index(label)
    lines[index].set_visible(not lines[index].get_visible())
    plt.draw()

check.on_clicked(toggle_species)
ax.legend()
plt.show()
```

## Integration with Other Libraries

### Pandas Integration

```python
# Direct pandas plotting with seaborn style
sns.set_theme()
tips.plot(x='total_bill', y='tip', kind='scatter')

# Using pandas groupby with seaborn
grouped_data = tips.groupby(['day', 'time'])['total_bill'].mean().reset_index()
sns.barplot(data=grouped_data, x='day', y='total_bill', hue='time')

# Melting data for seaborn
tips_long = pd.melt(tips, id_vars=['time', 'day'], 
                   value_vars=['total_bill', 'tip'])
sns.boxplot(data=tips_long, x='variable', y='value', hue='time')
```

### Statistical Testing Integration

```python
from scipy.stats import ttest_ind
import statannotations.stats as stats_annotations

# Statistical annotations on plots
ax = sns.boxplot(data=tips, x='time', y='total_bill')

# Add significance testing
box_pairs = [('Lunch', 'Dinner')]
annotator = stats_annotations.Annotator(ax, box_pairs, data=tips, 
                                       x='time', y='total_bill')
annotator.configure(test='t-test_ind', text_format='star')
annotator.apply_and_annotate()
```

### Machine Learning Integration

```python
from sklearn.datasets import load_digits
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

# Load data
digits = load_digits()
X, y = digits.data, digits.target

# PCA
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)

# K-means clustering
kmeans = KMeans(n_clusters=10, random_state=42)
clusters = kmeans.fit_predict(X)

# Create DataFrame for seaborn
df_ml = pd.DataFrame({
    'PC1': X_pca[:, 0],
    'PC2': X_pca[:, 1],
    'True_Label': y,
    'Cluster': clusters
})

# Visualize
fig, axes = plt.subplots(1, 2, figsize=(15, 6))

sns.scatterplot(data=df_ml, x='PC1', y='PC2', hue='True_Label', 
                palette='tab10', ax=axes[0])
axes[0].set_title('True Labels')

sns.scatterplot(data=df_ml, x='PC1', y='PC2', hue='Cluster', 
                palette='tab10', ax=axes[1])
axes[1].set_title('K-means Clusters')

plt.tight_layout()
```

## Common Use Cases

### Exploratory Data Analysis

```python
def explore_dataset(df, target_column=None):
    """Comprehensive EDA function using seaborn"""
    
    # Dataset overview
    print(f"Dataset shape: {df.shape}")
    print(f"Missing values:\n{df.isnull().sum()}")
    
    # Numeric columns
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    
    # Categorical columns
    cat_cols = df.select_dtypes(include=['object', 'category']).columns
    
    # Correlation heatmap
    if len(numeric_cols) > 1:
        plt.figure(figsize=(10, 8))
        sns.heatmap(df[numeric_cols].corr(), annot=True, cmap='coolwarm', center=0)
        plt.title('Correlation Matrix')
        plt.tight_layout()
        plt.show()
    
    # Distribution plots
    if len(numeric_cols) > 0:
        n_cols = min(3, len(numeric_cols))
        n_rows = (len(numeric_cols) + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5*n_rows))
        axes = axes.flatten() if n_rows > 1 else [axes]
        
        for i, col in enumerate(numeric_cols):
            if target_column and target_column in df.columns:
                sns.histplot(data=df, x=col, hue=target_column, ax=axes[i], kde=True)
            else:
                sns.histplot(data=df, x=col, ax=axes[i], kde=True)
            axes[i].set_title(f'Distribution of {col}')
        
        plt.tight_layout()
        plt.show()
    
    # Pairplot
    if len(numeric_cols) > 1 and len(numeric_cols) <= 6:
        if target_column and target_column in df.columns:
            sns.pairplot(df, vars=numeric_cols, hue=target_column)
        else:
            sns.pairplot(df[numeric_cols])
        plt.show()

# Usage
explore_dataset(tips, target_column='time')
```

### Time Series Visualization

```python
# Prepare time series data
np.random.seed(42)
dates = pd.date_range('2020-01-01', periods=365, freq='D')
values = np.cumsum(np.random.randn(365)) + 100
ts_data = pd.DataFrame({'date': dates, 'value': values})
ts_data['month'] = ts_data['date'].dt.month
ts_data['day_of_week'] = ts_data['date'].dt.day_name()

# Time series plots
fig, axes = plt.subplots(3, 1, figsize=(15, 12))

# Line plot
sns.lineplot(data=ts_data, x='date', y='value', ax=axes[0])
axes[0].set_title('Time Series')

# Monthly boxplot
sns.boxplot(data=ts_data, x='month', y='value', ax=axes[1])
axes[1].set_title('Monthly Distribution')

# Day of week pattern
day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
sns.boxplot(data=ts_data, x='day_of_week', y='value', order=day_order, ax=axes[2])
axes[2].set_title('Day of Week Pattern')
axes[2].tick_params(axis='x', rotation=45)

plt.tight_layout()
plt.show()
```

### Scientific Publication Plots

```python
def publication_plot():
    """Create publication-ready plots"""
    
    # Set publication style
    sns.set_theme(style="white", context="paper", font_scale=1.2)
    
    # Create figure with specific size (for journal requirements)
    fig = plt.figure(figsize=(8.5, 11))  # US Letter size
    
    # Multiple subplots
    gs = fig.add_gridspec(3, 2, height_ratios=[1, 1, 1.2])
    
    # Plot A: Scatter with regression
    ax1 = fig.add_subplot(gs[0, 0])
    sns.scatterplot(data=tips, x="total_bill", y="tip", alpha=0.6, ax=ax1)
    sns.regplot(data=tips, x="total_bill", y="tip", scatter=False, 
                color='red', ax=ax1)
    ax1.set_title('A', fontweight='bold', loc='left')
    ax1.set_xlabel('Total Bill ($)')
    ax1.set_ylabel('Tip ($)')
    
    # Plot B: Box plot with statistics
    ax2 = fig.add_subplot(gs[0, 1])
    sns.boxplot(data=tips, x="time", y="total_bill", ax=ax2)
    ax2.set_title('B', fontweight='bold', loc='left')
    ax2.set_xlabel('Time')
    ax2.set_ylabel('Total Bill ($)')
    
    # Plot C: Violin plot
    ax3 = fig.add_subplot(gs[1, :])
    sns.violinplot(data=tips, x="day", y="total_bill", hue="time", ax=ax3)
    ax3.set_title('C', fontweight='bold', loc='left')
    ax3.set_xlabel('Day of Week')
    ax3.set_ylabel('Total Bill ($)')
    
    # Plot D: Correlation heatmap
    ax4 = fig.add_subplot(gs[2, :])
    corr = tips.select_dtypes(include=[np.number]).corr()
    sns.heatmap(corr, annot=True, cmap='RdBu_r', center=0, ax=ax4,
                square=True, fmt='.2f')
    ax4.set_title('D', fontweight='bold', loc='left')
    
    # Remove spines for cleaner look
    for ax in [ax1, ax2, ax3]:
        sns.despine(ax=ax)
    
    plt.tight_layout()
    return fig

# Create and save publication plot
fig = publication_plot()
fig.savefig('publication_plot.pdf', dpi=300, bbox_inches='tight', 
            facecolor='white', edgecolor='none')
plt.show()
```

## Best Practices

### Performance Tips

```python
# Use appropriate figure sizes
sns.set_context("notebook")  # Instead of making everything larger

# Efficient color palettes
# Good: Use built-in palettes
sns.set_palette("husl")

# Avoid: Creating custom palettes repeatedly in loops
# for i in range(100):
#     custom_palette = ["#FF6B6B", "#4ECDC4", "#45B7D1"]  # Inefficient

# Batch processing for multiple plots
def create_multiple_plots(data, columns):
    """Efficiently create multiple plots"""
    n_cols = len(columns)
    fig, axes = plt.subplots(1, n_cols, figsize=(5*n_cols, 5))
    
    for i, col in enumerate(columns):
        ax = axes[i] if n_cols > 1 else axes
        sns.histplot(data=data, x=col, ax=ax)
        ax.set_title(col)
    
    plt.tight_layout()
    return fig
```

### Memory Management

```python
# For large datasets, sample data
def plot_large_dataset(df, sample_size=10000):
    """Handle large datasets efficiently"""
    if len(df) > sample_size:
        df_sample = df.sample(n=sample_size, random_state=42)
        print(f"Sampling {sample_size} rows from {len(df)} total rows")
    else:
        df_sample = df
    
    return sns.scatterplot(data=df_sample, x='x', y='y')

# Close figures to free memory
plt.close('all')  # Close all figures
plt.close(fig)    # Close specific figure
```

### Aesthetic Consistency

```python
# Create consistent style function
def set_publication_style():
    """Set consistent publication-ready style"""
    sns.set_theme(
        style="white",
        context="paper",
        font_scale=1.2,
        rc={
            "axes.spines.right": False,
            "axes.spines.top": False,
            "axes.grid": True,
            "axes.grid.alpha": 0.3,
            "figure.facecolor": "white",
            "axes.facecolor": "white"
        }
    )

# Use consistent color schemes
COLORS = {
    'primary': '#1f77b4',
    'secondary': '#ff7f0e',
    'success': '#2ca02c',
    'danger': '#d62728',
    'warning': '#ff7f0e',
    'info': '#17a2b8',
    'light': '#f8f9fa',
    'dark': '#343a40'
}

# Apply consistent colors
sns.scatterplot(data=tips, x="total_bill", y="tip", color=COLORS['primary'])
```

## Troubleshooting Common Issues

### Data Format Issues

```python
# Ensure proper data types
def prepare_data_for_seaborn(df):
    """Prepare DataFrame for seaborn plotting"""
    df = df.copy()
    
    # Convert categorical variables
    for col in df.select_dtypes(include=['object']).columns:
        if df[col].nunique() < 10:  # Arbitrary threshold
            df[col] = df[col].astype('category')
    
    # Handle datetime columns
    datetime_cols = df.select_dtypes(include=['datetime64']).columns
    for col in datetime_cols:
        df[f'{col}_year'] = df[col].dt.year
        df[f'{col}_month'] = df[col].dt.month
    
    return df

# Handle missing values
def handle_missing_data(df, strategy='drop'):
    """Handle missing data for plotting"""
    if strategy == 'drop':
        return df.dropna()
    elif strategy == 'fill_numeric':
        df_clean = df.copy()
        numeric_cols = df_clean.select_dtypes(include=[np.number]).columns
        df_clean[numeric_cols] = df_clean[numeric_cols].fillna(df_clean[numeric_cols].mean())
        return df_clean
    return df
```

### Plot Customization Issues

```python
# Fix overlapping labels
def fix_overlapping_labels(ax):
    """Fix common label overlap issues"""
    plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
    plt.tight_layout()

# Handle legend issues
def fix_legend_issues(ax, title=None, loc='best'):
    """Standardize legend appearance"""
    legend = ax.legend(title=title, loc=loc, frameon=True, 
                      fancybox=True, shadow=True)
    legend.get_frame().set_facecolor('white')
    legend.get_frame().set_alpha(0.9)

# Consistent axis formatting
def format_axes(ax, xlabel=None, ylabel=None, title=None):
    """Apply consistent axis formatting"""
    if xlabel:
        ax.set_xlabel(xlabel, fontweight='bold')
    if ylabel:
        ax.set_ylabel(ylabel, fontweight='bold')
    if title:
        ax.set_title(title, fontweight='bold', pad=20)
    
    # Format tick labels
    ax.tick_params(axis='both', which='major', labelsize=10)
    
    return ax
```

## Quick Reference

### Essential Functions

| Function | Purpose | Example |
|----------|---------|---------|
| `sns.scatterplot()` | Scatter plot | `sns.scatterplot(data=df, x='x', y='y', hue='category')` |
| `sns.lineplot()` | Line plot | `sns.lineplot(data=df, x='time', y='value')` |
| `sns.histplot()` | Histogram | `sns.histplot(data=df, x='values', hue='group')` |
| `sns.boxplot()` | Box plot | `sns.boxplot(data=df, x='category', y='values')` |
| `sns.heatmap()` | Heat map | `sns.heatmap(df.corr(), annot=True)` |
| `sns.pairplot()` | Pair plot | `sns.pairplot(data=df, hue='species')` |

### Figure-Level Functions

| Function | Axes-Level Equivalent | Use Case |
|----------|----------------------|----------|
| `sns.relplot()` | `sns.scatterplot()`, `sns.lineplot()` | Relationships with faceting |
| `sns.displot()` | `sns.histplot()`, `sns.kdeplot()`, `sns.ecdfplot()` | Distributions with faceting |
| `sns.catplot()` | `sns.boxplot()`, `sns.violinplot()`, etc. | Categories with faceting |
| `sns.lmplot()` | `sns.regplot()` | Linear models with faceting |

### Common Parameters

| Parameter | Purpose | Values |
|-----------|---------|---------|
| `data` | DataFrame | pandas DataFrame |
| `x`, `y` | Variables to plot | Column names |
| `hue` | Grouping variable (color) | Column name |
| `style` | Grouping variable (style) | Column name |
| `size` | Grouping variable (size) | Column name |
| `col`, `row` | Faceting variables | Column names |
| `palette` | Color palette | 'viridis', 'Set1', custom list |
| `alpha` | Transparency | 0.0 to 1.0 |

### Color Palettes

| Type | Examples | Use Case |
|------|----------|----------|
| Qualitative | 'Set1', 'tab10', 'husl' | Categorical data |
| Sequential | 'viridis', 'Blues', 'rocket' | Ordered data |
| Diverging | 'RdBu', 'coolwarm', 'vlag' | Data with meaningful center |

### Styling Contexts

| Context | Use Case | Relative Size |
|---------|----------|---------------|
| `paper` | Journal figures | Smallest |
| `notebook` | Jupyter notebooks | Default |
| `talk` | Presentations | Larger |
| `poster` | Conference posters | Largest |