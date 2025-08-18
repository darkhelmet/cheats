# SciPy Cheat Sheet

## Installation

```bash
# Basic installation
pip install scipy

# With all optional dependencies
pip install scipy[all]

# Check version
python -c "import scipy; print(scipy.__version__)"

# Check available submodules
python -c "import scipy; print([x for x in dir(scipy) if not x.startswith('_')])"
```

## Basic Setup

```python
# Essential imports
import numpy as np
import scipy as sp
from scipy import stats, optimize, integrate, linalg, signal, interpolate

# Common import pattern for specific modules
from scipy.stats import norm, t, chi2, pearsonr
from scipy.optimize import minimize, curve_fit
from scipy.signal import find_peaks, periodogram
from scipy.interpolate import interp1d
```

## Core Functionality

### Statistics Module (`scipy.stats`)

#### Continuous Distributions

```python
from scipy.stats import norm, t, chi2, f, uniform, expon, gamma, beta

# Normal distribution
mean, std = 0, 1
x = np.linspace(-3, 3, 100)

# Probability density function (PDF)
pdf_values = norm.pdf(x, mean, std)

# Cumulative distribution function (CDF)
cdf_values = norm.cdf(x, mean, std)

# Percent point function (inverse CDF)
percentiles = norm.ppf([0.05, 0.5, 0.95], mean, std)

# Random variates
samples = norm.rvs(mean, std, size=1000)

# Summary statistics
print(f"Mean: {norm.mean(mean, std)}")
print(f"Variance: {norm.var(mean, std)}")
print(f"Standard deviation: {norm.std(mean, std)}")
```

#### Discrete Distributions

```python
from scipy.stats import binom, poisson, hypergeom, geom

# Binomial distribution
n, p = 10, 0.3
k = np.arange(0, 11)

# Probability mass function (PMF)
pmf_values = binom.pmf(k, n, p)

# CDF for discrete distributions
cdf_values = binom.cdf(k, n, p)

# Random samples
samples = binom.rvs(n, p, size=1000)

# Poisson distribution
lam = 3.5  # lambda parameter
k = np.arange(0, 15)
pmf_poisson = poisson.pmf(k, lam)
```

#### Statistical Tests

```python
from scipy.stats import ttest_1samp, ttest_ind, ttest_rel, chi2_contingency
from scipy.stats import pearsonr, spearmanr, kendalltau, mannwhitneyu

# One-sample t-test
data = np.random.normal(5, 2, 100)
statistic, p_value = ttest_1samp(data, popmean=4.5)
print(f"t-statistic: {statistic:.4f}, p-value: {p_value:.4f}")

# Two-sample t-test (independent)
group1 = np.random.normal(5, 2, 50)
group2 = np.random.normal(5.5, 2, 50)
stat, p = ttest_ind(group1, group2)

# Paired t-test
before = np.random.normal(100, 15, 30)
after = before + np.random.normal(2, 5, 30)
stat, p = ttest_rel(before, after)

# Correlation tests
x = np.random.normal(0, 1, 100)
y = 2*x + np.random.normal(0, 0.5, 100)

# Pearson correlation
corr_pearson, p_pearson = pearsonr(x, y)

# Spearman rank correlation
corr_spearman, p_spearman = spearmanr(x, y)

# Kendall's tau
tau, p_kendall = kendalltau(x, y)

# Mann-Whitney U test (non-parametric)
stat, p = mannwhitneyu(group1, group2, alternative='two-sided')
```

#### Chi-square Tests

```python
from scipy.stats import chi2_contingency, chisquare

# Chi-square test of independence
contingency_table = np.array([[10, 20, 30],
                              [15, 25, 35],
                              [20, 30, 40]])
chi2_stat, p_value, dof, expected = chi2_contingency(contingency_table)

# Goodness of fit test
observed = [16, 18, 16, 14, 12, 12]
expected = [16, 16, 16, 16, 16, 16]
chi2_stat, p_value = chisquare(observed, expected)
```

### Optimization Module (`scipy.optimize`)

#### Function Minimization

```python
from scipy.optimize import minimize, minimize_scalar, differential_evolution

# Scalar function minimization
def f(x):
    return (x - 2)**2

result = minimize_scalar(f)
print(f"Minimum at x = {result.x:.4f}, f(x) = {result.fun:.4f}")

# Multivariable function minimization
def rosenbrock(x):
    return 100*(x[1] - x[0]**2)**2 + (1 - x[0])**2

# Initial guess
x0 = [0, 0]

# Minimize using different methods
result_bfgs = minimize(rosenbrock, x0, method='BFGS')
result_nm = minimize(rosenbrock, x0, method='Nelder-Mead')

print(f"BFGS result: {result_bfgs.x}")
print(f"Nelder-Mead result: {result_nm.x}")

# With bounds
bounds = [(0, 2), (0, 2)]
result_bounded = minimize(rosenbrock, x0, method='L-BFGS-B', bounds=bounds)

# Global optimization
result_global = differential_evolution(rosenbrock, bounds)
```

#### Root Finding

```python
from scipy.optimize import root, fsolve, brentq

# Single variable root finding
def equation(x):
    return x**3 - 2*x - 5

# Brent's method (requires bracketing interval)
root_brent = brentq(equation, 2, 3)

# Multi-variable root finding
def system_equations(vars):
    x, y = vars
    eq1 = x**2 + y**2 - 1
    eq2 = x - y**2
    return [eq1, eq2]

# Initial guess
initial_guess = [0.5, 0.5]
solution = fsolve(system_equations, initial_guess)

# Using root function (more options)
sol = root(system_equations, initial_guess, method='hybr')
```

#### Curve Fitting

```python
from scipy.optimize import curve_fit

# Define model function
def exponential_model(x, a, b, c):
    return a * np.exp(b * x) + c

# Generate sample data with noise
x_data = np.linspace(0, 4, 50)
y_true = exponential_model(x_data, 2.5, 1.3, 0.5)
y_data = y_true + 0.2 * np.random.normal(size=len(x_data))

# Fit the curve
popt, pcov = curve_fit(exponential_model, x_data, y_data)
a_fit, b_fit, c_fit = popt

# Parameter uncertainties
param_errors = np.sqrt(np.diag(pcov))

print(f"Fitted parameters: a={a_fit:.3f}±{param_errors[0]:.3f}, "
      f"b={b_fit:.3f}±{param_errors[1]:.3f}, c={c_fit:.3f}±{param_errors[2]:.3f}")

# Plot results
y_fit = exponential_model(x_data, *popt)
import matplotlib.pyplot as plt
plt.plot(x_data, y_data, 'o', label='Data')
plt.plot(x_data, y_fit, '-', label='Fitted curve')
plt.legend()
```

### Linear Algebra Module (`scipy.linalg`)

#### Basic Operations

```python
from scipy.linalg import solve, det, inv, eig, svd, norm, cholesky

# System of linear equations Ax = b
A = np.array([[3, 2, -1],
              [2, -2, 4],
              [-1, 0.5, -1]])
b = np.array([1, -2, 0])

# Solve the system
x = solve(A, b)
print(f"Solution: {x}")

# Matrix determinant
determinant = det(A)

# Matrix inverse
A_inv = inv(A)

# Verify: A * A_inv should be identity
identity_check = np.allclose(A @ A_inv, np.eye(3))
```

#### Eigenvalues and Eigenvectors

```python
# Eigendecomposition
eigenvalues, eigenvectors = eig(A)
print(f"Eigenvalues: {eigenvalues}")

# Singular Value Decomposition
U, s, Vt = svd(A)
print(f"Singular values: {s}")

# Matrix norms
frobenius_norm = norm(A, 'fro')
spectral_norm = norm(A, 2)
l1_norm = norm(A, 1)
```

#### Decompositions

```python
from scipy.linalg import lu, qr, cholesky, schur

# LU decomposition
P, L, U = lu(A)

# QR decomposition
Q, R = qr(A)

# Cholesky decomposition (for positive definite matrices)
# Create a positive definite matrix
A_pd = A.T @ A
L_chol = cholesky(A_pd, lower=True)

# Schur decomposition
T, Z = schur(A)
```

### Signal Processing Module (`scipy.signal`)

#### Filtering

```python
from scipy.signal import butter, filtfilt, savgol_filter, medfilt

# Create sample signal
fs = 1000  # Sampling frequency
t = np.linspace(0, 1, fs, endpoint=False)
signal = np.sin(2*np.pi*5*t) + 0.5*np.sin(2*np.pi*25*t) + np.random.normal(0, 0.1, len(t))

# Butterworth filter
def butter_lowpass_filter(data, cutoff, fs, order=4):
    nyquist = 0.5 * fs
    normal_cutoff = cutoff / nyquist
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    filtered_data = filtfilt(b, a, data)
    return filtered_data

# Apply low-pass filter
cutoff_freq = 10  # Hz
filtered_signal = butter_lowpass_filter(signal, cutoff_freq, fs)

# Savitzky-Golay filter (good for smoothing)
window_length = 51  # Must be odd
poly_order = 3
smoothed_signal = savgol_filter(signal, window_length, poly_order)

# Median filter (good for removing spikes)
kernel_size = 5
median_filtered = medfilt(signal, kernel_size)
```

#### Spectral Analysis

```python
from scipy.signal import periodogram, welch, spectrogram
from scipy.fft import fft, fftfreq

# Periodogram (power spectral density)
frequencies, psd = periodogram(signal, fs)

# Welch's method (averaged periodogram)
f_welch, psd_welch = welch(signal, fs, nperseg=256)

# Spectrogram (time-frequency analysis)
f_spec, t_spec, Sxx = spectrogram(signal, fs, nperseg=256)

# FFT
fft_values = fft(signal)
fft_freq = fftfreq(len(signal), 1/fs)

# Plot spectrum
import matplotlib.pyplot as plt
plt.figure(figsize=(12, 4))

plt.subplot(131)
plt.plot(t[:500], signal[:500])
plt.title('Time Domain')
plt.xlabel('Time (s)')

plt.subplot(132)
plt.semilogy(frequencies, psd)
plt.title('Power Spectral Density')
plt.xlabel('Frequency (Hz)')

plt.subplot(133)
plt.pcolormesh(t_spec, f_spec, 10*np.log10(Sxx), shading='gouraud')
plt.title('Spectrogram')
plt.ylabel('Frequency (Hz)')
plt.xlabel('Time (s)')
plt.tight_layout()
```

#### Peak Finding

```python
from scipy.signal import find_peaks, peak_widths, peak_prominences

# Generate signal with peaks
x = np.linspace(0, 10, 1000)
signal_peaks = np.sin(x) + 0.5*np.sin(3*x) + 0.2*np.random.randn(1000)

# Find peaks
peaks, properties = find_peaks(signal_peaks, height=0.5, distance=20)

# Peak properties
widths = peak_widths(signal_peaks, peaks, rel_height=0.5)
prominences = peak_prominences(signal_peaks, peaks)

# Plot results
plt.figure(figsize=(12, 6))
plt.plot(x, signal_peaks, label='Signal')
plt.plot(x[peaks], signal_peaks[peaks], 'ro', label='Peaks')
plt.legend()
```

### Integration Module (`scipy.integrate`)

#### Numerical Integration

```python
from scipy.integrate import quad, dblquad, tplquad, odeint, solve_ivp

# Single integral
def integrand(x):
    return np.exp(-x**2)

# Integrate from 0 to infinity
result, error = quad(integrand, 0, np.inf)
print(f"∫₀^∞ e^(-x²) dx = {result:.6f} ± {error:.2e}")

# Double integral
def integrand_2d(y, x):
    return x*y**2

# Integrate over rectangle [0,1] × [0,2]
result_2d, error_2d = dblquad(integrand_2d, 0, 1, lambda x: 0, lambda x: 2)

# Triple integral
def integrand_3d(z, y, x):
    return x*y*z

result_3d, error_3d = tplquad(integrand_3d, 0, 1, lambda x: 0, lambda x: 1, 
                              lambda x, y: 0, lambda x, y: 1)
```

#### Ordinary Differential Equations (ODEs)

```python
from scipy.integrate import odeint, solve_ivp

# Solve dy/dt = -2y with initial condition y(0) = 1
def dydt(y, t):
    return -2*y

t = np.linspace(0, 2, 100)
y0 = 1
solution = odeint(dydt, y0, t)

# System of ODEs: Lotka-Volterra (predator-prey)
def lotka_volterra(t, z, a, b, c, d):
    x, y = z
    return [a*x - b*x*y, -c*y + d*x*y]

# Parameters
a, b, c, d = 1.0, 0.1, 1.5, 0.075
initial_conditions = [10, 5]  # [prey, predator]
t_span = (0, 15)
t_eval = np.linspace(0, 15, 1000)

# Solve using solve_ivp (more modern interface)
sol = solve_ivp(lotka_volterra, t_span, initial_conditions, 
                t_eval=t_eval, args=(a, b, c, d), dense_output=True)

# Plot phase portrait
plt.figure(figsize=(12, 5))
plt.subplot(121)
plt.plot(sol.t, sol.y[0], label='Prey')
plt.plot(sol.t, sol.y[1], label='Predator')
plt.xlabel('Time')
plt.ylabel('Population')
plt.legend()

plt.subplot(122)
plt.plot(sol.y[0], sol.y[1])
plt.xlabel('Prey')
plt.ylabel('Predator')
plt.title('Phase Portrait')
```

### Interpolation Module (`scipy.interpolate`)

#### 1D Interpolation

```python
from scipy.interpolate import interp1d, UnivariateSpline, CubicSpline

# Sample data
x = np.linspace(0, 10, 11)
y = np.sin(x)

# Different interpolation methods
f_linear = interp1d(x, y, kind='linear')
f_cubic = interp1d(x, y, kind='cubic')
f_spline = UnivariateSpline(x, y, s=0)  # s=0 for interpolation
f_cubic_spline = CubicSpline(x, y)

# Evaluate at new points
x_new = np.linspace(0, 10, 100)
y_linear = f_linear(x_new)
y_cubic = f_cubic(x_new)
y_spline = f_spline(x_new)
y_cubic_spline = f_cubic_spline(x_new)

# Plot comparison
plt.figure(figsize=(10, 6))
plt.plot(x, y, 'o', label='Data points')
plt.plot(x_new, y_linear, '-', label='Linear')
plt.plot(x_new, y_cubic, '--', label='Cubic')
plt.plot(x_new, y_spline, '-.', label='Spline')
plt.plot(x_new, y_cubic_spline, ':', label='Cubic Spline')
plt.legend()
plt.title('Interpolation Methods Comparison')
```

#### 2D Interpolation

```python
from scipy.interpolate import griddata, RegularGridInterpolator

# Scattered data interpolation
np.random.seed(42)
points = np.random.rand(100, 2) * 10
values = np.sin(points[:, 0]) * np.cos(points[:, 1])

# Create regular grid
xi = np.linspace(0, 10, 50)
yi = np.linspace(0, 10, 50)
xi_grid, yi_grid = np.meshgrid(xi, yi)

# Interpolate using different methods
zi_nearest = griddata(points, values, (xi_grid, yi_grid), method='nearest')
zi_linear = griddata(points, values, (xi_grid, yi_grid), method='linear')
zi_cubic = griddata(points, values, (xi_grid, yi_grid), method='cubic')

# Regular grid interpolation (faster for regular grids)
x_reg = np.linspace(0, 10, 20)
y_reg = np.linspace(0, 10, 20)
X_reg, Y_reg = np.meshgrid(x_reg, y_reg)
Z_reg = np.sin(X_reg) * np.cos(Y_reg)

interp_func = RegularGridInterpolator((x_reg, y_reg), Z_reg)
new_points = np.column_stack([xi_grid.ravel(), yi_grid.ravel()])
zi_reg = interp_func(new_points).reshape(xi_grid.shape)
```

## Advanced Features

### Spatial Algorithms (`scipy.spatial`)

```python
from scipy.spatial import distance, KDTree, ConvexHull, Voronoi
from scipy.spatial.distance import pdist, squareform

# Distance calculations
points = np.random.rand(10, 2)

# Pairwise distances
distances = pdist(points)
distance_matrix = squareform(distances)

# Different distance metrics
euclidean_dist = pdist(points, metric='euclidean')
manhattan_dist = pdist(points, metric='manhattan')
cosine_dist = pdist(points, metric='cosine')

# KD-Tree for nearest neighbor searches
tree = KDTree(points)

# Find k nearest neighbors
query_point = [0.5, 0.5]
distances, indices = tree.query(query_point, k=3)

# Convex Hull
hull = ConvexHull(points)
hull_points = points[hull.vertices]

# Voronoi diagram
vor = Voronoi(points)
```

### Sparse Matrices (`scipy.sparse`)

```python
from scipy.sparse import csr_matrix, csc_matrix, diags, eye
from scipy.sparse.linalg import spsolve

# Create sparse matrix
row = np.array([0, 0, 1, 2, 2, 2])
col = np.array([0, 2, 1, 0, 1, 2])
data = np.array([1, 2, 3, 4, 5, 6])

# Compressed Sparse Row matrix
sparse_matrix = csr_matrix((data, (row, col)), shape=(3, 3))

# Create diagonal sparse matrix
diag_sparse = diags([1, 2, 3], offsets=[0], shape=(3, 3))

# Sparse identity matrix
sparse_eye = eye(1000, format='csr')

# Solve sparse linear system
A_sparse = csr_matrix([[1, 2], [3, 4]])
b = np.array([1, 2])
x = spsolve(A_sparse, b)
```

### FFT Module (`scipy.fft`)

```python
from scipy.fft import fft, ifft, fft2, ifft2, fftfreq, rfft

# 1D FFT
signal = np.sin(2*np.pi*5*np.linspace(0, 1, 1000))
fft_signal = fft(signal)
freq = fftfreq(len(signal), 1/1000)

# Real FFT (for real-valued signals)
rfft_signal = rfft(signal)
rfreq = fftfreq(len(signal), 1/1000)[:len(rfft_signal)]

# 2D FFT
image = np.random.rand(100, 100)
fft2_image = fft2(image)
ifft2_image = ifft2(fft2_image)  # Should recover original

# Verify reconstruction
reconstruction_error = np.max(np.abs(image - ifft2_image.real))
print(f"Reconstruction error: {reconstruction_error:.2e}")
```

## Common Use Cases

### Statistical Analysis Workflow

```python
# Complete statistical analysis example
import pandas as pd
from scipy import stats

# Generate sample data
np.random.seed(42)
group_a = np.random.normal(100, 15, 50)
group_b = np.random.normal(105, 15, 50)

# Descriptive statistics
print("Group A:")
print(f"Mean: {np.mean(group_a):.2f}, Std: {np.std(group_a, ddof=1):.2f}")
print("Group B:")
print(f"Mean: {np.mean(group_b):.2f}, Std: {np.std(group_b, ddof=1):.2f}")

# Test for normality
_, p_a = stats.shapiro(group_a)
_, p_b = stats.shapiro(group_b)
print(f"\nNormality tests (Shapiro-Wilk):")
print(f"Group A p-value: {p_a:.4f}")
print(f"Group B p-value: {p_b:.4f}")

# Test for equal variances
_, p_levene = stats.levene(group_a, group_b)
print(f"\nEqual variances test (Levene): p-value = {p_levene:.4f}")

# Choose appropriate t-test
if p_levene > 0.05:  # Equal variances
    t_stat, p_ttest = stats.ttest_ind(group_a, group_b, equal_var=True)
    print(f"\nStudent's t-test: t = {t_stat:.4f}, p = {p_ttest:.4f}")
else:  # Unequal variances
    t_stat, p_ttest = stats.ttest_ind(group_a, group_b, equal_var=False)
    print(f"\nWelch's t-test: t = {t_stat:.4f}, p = {p_ttest:.4f}")

# Effect size (Cohen's d)
pooled_std = np.sqrt(((len(group_a)-1)*np.var(group_a, ddof=1) + 
                      (len(group_b)-1)*np.var(group_b, ddof=1)) / 
                     (len(group_a) + len(group_b) - 2))
cohens_d = (np.mean(group_a) - np.mean(group_b)) / pooled_std
print(f"Cohen's d: {cohens_d:.4f}")
```

### Signal Processing Pipeline

```python
# Complete signal processing example
from scipy import signal
import matplotlib.pyplot as plt

# Generate noisy signal
fs = 1000
t = np.linspace(0, 2, 2*fs, endpoint=False)
clean_signal = (np.sin(2*np.pi*10*t) + 
                0.5*np.sin(2*np.pi*20*t) + 
                0.3*np.sin(2*np.pi*30*t))
noisy_signal = clean_signal + 0.2*np.random.randn(len(t))

# Design and apply filters
# Low-pass filter
sos_low = signal.butter(4, 25, btype='low', fs=fs, output='sos')
filtered_low = signal.sosfiltfilt(sos_low, noisy_signal)

# Band-pass filter
sos_band = signal.butter(4, [15, 25], btype='band', fs=fs, output='sos')
filtered_band = signal.sosfiltfilt(sos_band, noisy_signal)

# Spectral analysis
f, psd = signal.welch(noisy_signal, fs, nperseg=1024)
f_clean, psd_clean = signal.welch(clean_signal, fs, nperseg=1024)
f_filt, psd_filt = signal.welch(filtered_low, fs, nperseg=1024)

# Find peaks in filtered signal
peaks, properties = signal.find_peaks(filtered_low, height=0.5, distance=50)

# Plot results
fig, axes = plt.subplots(2, 2, figsize=(15, 10))

# Time domain
axes[0,0].plot(t[:500], noisy_signal[:500], alpha=0.7, label='Noisy')
axes[0,0].plot(t[:500], filtered_low[:500], label='Filtered')
axes[0,0].set_title('Time Domain')
axes[0,0].legend()

# Frequency domain
axes[0,1].semilogy(f, psd, alpha=0.7, label='Noisy')
axes[0,1].semilogy(f_filt, psd_filt, label='Filtered')
axes[0,1].set_title('Power Spectral Density')
axes[0,1].legend()

# Peak detection
axes[1,0].plot(filtered_low[:1000])
axes[1,0].plot(peaks[peaks<1000], filtered_low[peaks[peaks<1000]], 'ro')
axes[1,0].set_title('Peak Detection')

# Spectrogram
f_spec, t_spec, Sxx = signal.spectrogram(noisy_signal, fs, nperseg=256)
axes[1,1].pcolormesh(t_spec, f_spec, 10*np.log10(Sxx), shading='gouraud')
axes[1,1].set_title('Spectrogram')
axes[1,1].set_ylabel('Frequency (Hz)')
axes[1,1].set_xlabel('Time (s)')

plt.tight_layout()
```

### Optimization Problem

```python
# Multi-modal optimization example
from scipy.optimize import minimize, differential_evolution, basinhopping

# Define a complex function with multiple local minima
def complex_function(x):
    x1, x2 = x
    return (x1**2 + x2 - 11)**2 + (x1 + x2**2 - 7)**2

# Local optimization (may get stuck in local minima)
x0 = [0, 0]
result_local = minimize(complex_function, x0, method='BFGS')

# Global optimization methods
bounds = [(-5, 5), (-5, 5)]
result_global = differential_evolution(complex_function, bounds, seed=42)

# Basin-hopping (another global method)
minimizer_kwargs = {"method": "BFGS"}
result_basinhopping = basinhopping(complex_function, x0, 
                                   minimizer_kwargs=minimizer_kwargs, 
                                   niter=100, T=1.0, stepsize=0.5)

print("Local minimum (BFGS):", result_local.x, "f =", result_local.fun)
print("Global minimum (DE):", result_global.x, "f =", result_global.fun)
print("Basin-hopping result:", result_basinhopping.x, "f =", result_basinhopping.fun)

# Visualize the function and minima
x1_range = np.linspace(-5, 5, 100)
x2_range = np.linspace(-5, 5, 100)
X1, X2 = np.meshgrid(x1_range, x2_range)
Z = complex_function([X1, X2])

plt.figure(figsize=(10, 8))
contour = plt.contour(X1, X2, Z, levels=50)
plt.colorbar(contour)
plt.plot(result_local.x[0], result_local.x[1], 'ro', markersize=10, label='Local min')
plt.plot(result_global.x[0], result_global.x[1], 'go', markersize=10, label='Global min')
plt.xlabel('x1')
plt.ylabel('x2')
plt.title('Function Landscape with Optimization Results')
plt.legend()
```

## Integration with Other Libraries

### NumPy Integration

```python
# SciPy functions work seamlessly with NumPy arrays
data = np.random.exponential(2, 1000)

# Fit distribution
params = stats.expon.fit(data)
print(f"Fitted parameters: scale = {params[1]:.4f}")

# Kolmogorov-Smirnov test
ks_stat, p_value = stats.kstest(data, lambda x: stats.expon.cdf(x, *params))
print(f"K-S test: statistic = {ks_stat:.4f}, p-value = {p_value:.4f}")
```

### Matplotlib Integration

```python
# Visualization with matplotlib
fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# Q-Q plot
stats.probplot(data, dist="expon", plot=axes[0,0])
axes[0,0].set_title("Q-Q Plot")

# Histogram with fitted PDF
axes[0,1].hist(data, bins=50, density=True, alpha=0.7, label='Data')
x_fit = np.linspace(0, np.max(data), 100)
axes[0,1].plot(x_fit, stats.expon.pdf(x_fit, *params), 'r-', lw=2, label='Fitted PDF')
axes[0,1].legend()
axes[0,1].set_title("Histogram with Fitted Distribution")

# CDF comparison
sorted_data = np.sort(data)
empirical_cdf = np.arange(1, len(data)+1) / len(data)
theoretical_cdf = stats.expon.cdf(sorted_data, *params)
axes[1,0].plot(sorted_data, empirical_cdf, label='Empirical CDF')
axes[1,0].plot(sorted_data, theoretical_cdf, 'r-', label='Theoretical CDF')
axes[1,0].legend()
axes[1,0].set_title("CDF Comparison")

plt.tight_layout()
```

## Best Practices

### Performance Tips

```python
# Use vectorized operations
# Good
x = np.linspace(0, 10, 1000000)
y = stats.norm.pdf(x, 0, 1)

# Avoid loops when possible
# Bad (slow)
# y = np.array([stats.norm.pdf(xi, 0, 1) for xi in x])

# Use appropriate data types
data_float32 = np.random.rand(10000).astype(np.float32)
data_float64 = np.random.rand(10000).astype(np.float64)

# Profile your code for bottlenecks
import cProfile
cProfile.run('stats.ttest_ind(group_a, group_b)')
```

### Memory Management

```python
# For large datasets, consider chunking
def process_large_dataset(data, chunk_size=10000):
    results = []
    for i in range(0, len(data), chunk_size):
        chunk = data[i:i+chunk_size]
        result = stats.describe(chunk)
        results.append(result)
    return results

# Use sparse matrices for sparse data
from scipy.sparse import csr_matrix
# Instead of dense matrix with many zeros
# dense = np.zeros((10000, 10000))
# Use sparse representation
sparse = csr_matrix((10000, 10000))
```

### Error Handling

```python
# Handle numerical issues
def safe_division(a, b):
    try:
        result = a / b
        if np.isnan(result) or np.isinf(result):
            return None
        return result
    except ZeroDivisionError:
        return None

# Check for convergence in optimization
result = minimize(rosenbrock, x0, method='BFGS')
if result.success:
    print(f"Optimization successful: {result.x}")
else:
    print(f"Optimization failed: {result.message}")
    
# Validate statistical test assumptions
def check_normality(data, alpha=0.05):
    statistic, p_value = stats.shapiro(data)
    is_normal = p_value > alpha
    return is_normal, p_value
```

## Quick Reference

### Key Modules

| Module | Purpose | Common Functions |
|--------|---------|------------------|
| `scipy.stats` | Statistics | `norm`, `ttest_ind`, `pearsonr`, `chi2_contingency` |
| `scipy.optimize` | Optimization | `minimize`, `curve_fit`, `fsolve` |
| `scipy.linalg` | Linear algebra | `solve`, `eig`, `svd`, `cholesky` |
| `scipy.signal` | Signal processing | `butter`, `filtfilt`, `find_peaks`, `periodogram` |
| `scipy.interpolate` | Interpolation | `interp1d`, `griddata`, `CubicSpline` |
| `scipy.integrate` | Integration | `quad`, `solve_ivp`, `odeint` |
| `scipy.spatial` | Spatial algorithms | `distance`, `KDTree`, `ConvexHull` |
| `scipy.sparse` | Sparse matrices | `csr_matrix`, `spsolve` |

### Statistical Tests Quick Guide

| Test | Function | Use Case |
|------|----------|----------|
| One-sample t-test | `ttest_1samp` | Compare sample mean to population mean |
| Two-sample t-test | `ttest_ind` | Compare two independent groups |
| Paired t-test | `ttest_rel` | Compare paired/dependent samples |
| Mann-Whitney U | `mannwhitneyu` | Non-parametric comparison of two groups |
| Wilcoxon signed-rank | `wilcoxon` | Non-parametric paired test |
| Chi-square | `chi2_contingency` | Test independence in contingency tables |
| Kolmogorov-Smirnov | `kstest` | Test goodness of fit to distribution |
| Shapiro-Wilk | `shapiro` | Test for normality |
| Levene's test | `levene` | Test for equal variances |

### Distribution Quick Reference

| Distribution | Function | Parameters |
|--------------|----------|------------|
| Normal | `norm` | `loc` (mean), `scale` (std) |
| Student's t | `t` | `df` (degrees of freedom) |
| Chi-square | `chi2` | `df` (degrees of freedom) |
| F-distribution | `f` | `dfn`, `dfd` (degrees of freedom) |
| Binomial | `binom` | `n` (trials), `p` (probability) |
| Poisson | `poisson` | `mu` (rate parameter) |
| Exponential | `expon` | `scale` (1/rate) |
| Uniform | `uniform` | `loc` (lower), `scale` (range) |