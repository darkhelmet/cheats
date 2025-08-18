# Scikit-learn

## Installation
```bash
pip install scikit-learn
# Or with conda
conda install -c conda-forge scikit-learn
```

## Import
```python
import sklearn
from sklearn import datasets, model_selection, preprocessing, metrics
import numpy as np
import pandas as pd
```

## Basic Workflow

### 1. Load/Create Data
```python
from sklearn.datasets import load_iris, make_classification, make_regression

# Built-in datasets
iris = load_iris()
X, y = iris.data, iris.target

# Generate synthetic data
X, y = make_classification(n_samples=1000, n_features=20, random_state=42)
X, y = make_regression(n_samples=1000, n_features=10, random_state=42)
```

### 2. Train-Test Split
```python
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
```

### 3. Preprocessing
```python
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder

# Scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Alternative scalers
minmax_scaler = MinMaxScaler()
X_normalized = minmax_scaler.fit_transform(X)
```

### 4. Model Training and Prediction
```python
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

# Classification
clf = RandomForestClassifier(random_state=42)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
y_proba = clf.predict_proba(X_test)  # probability estimates
```

## Classification Algorithms

### Logistic Regression
```python
from sklearn.linear_model import LogisticRegression

lr = LogisticRegression(random_state=42)
lr.fit(X_train, y_train)
y_pred = lr.predict(X_test)
```

### Random Forest
```python
from sklearn.ensemble import RandomForestClassifier

rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)
feature_importance = rf.feature_importances_
```

### Support Vector Machine
```python
from sklearn.svm import SVC

svm = SVC(kernel='rbf', random_state=42)
svm.fit(X_train, y_train)
y_pred = svm.predict(X_test)
```

### Gradient Boosting
```python
from sklearn.ensemble import GradientBoostingClassifier

gb = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, random_state=42)
gb.fit(X_train, y_train)
```

### Decision Trees
```python
from sklearn.tree import DecisionTreeClassifier

dt = DecisionTreeClassifier(max_depth=5, random_state=42)
dt.fit(X_train, y_train)
```

### Naive Bayes
```python
from sklearn.naive_bayes import GaussianNB, MultinomialNB

nb = GaussianNB()
nb.fit(X_train, y_train)
```

### k-Nearest Neighbors
```python
from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)
```

## Regression Algorithms

### Linear Regression
```python
from sklearn.linear_model import LinearRegression, Ridge, Lasso

# Simple linear regression
lr = LinearRegression()
lr.fit(X_train, y_train)

# Ridge regression (L2 regularization)
ridge = Ridge(alpha=1.0)
ridge.fit(X_train, y_train)

# Lasso regression (L1 regularization)
lasso = Lasso(alpha=0.1)
lasso.fit(X_train, y_train)
```

### Random Forest Regressor
```python
from sklearn.ensemble import RandomForestRegressor

rf_reg = RandomForestRegressor(n_estimators=100, random_state=42)
rf_reg.fit(X_train, y_train)
y_pred = rf_reg.predict(X_test)
```

### Support Vector Regression
```python
from sklearn.svm import SVR

svr = SVR(kernel='rbf', C=100, gamma=0.1)
svr.fit(X_train, y_train)
```

## Clustering

### K-Means
```python
from sklearn.cluster import KMeans

kmeans = KMeans(n_clusters=3, random_state=42)
cluster_labels = kmeans.fit_predict(X)
centroids = kmeans.cluster_centers_
```

### Hierarchical Clustering
```python
from sklearn.cluster import AgglomerativeClustering

hierarchical = AgglomerativeClustering(n_clusters=3)
cluster_labels = hierarchical.fit_predict(X)
```

### DBSCAN
```python
from sklearn.cluster import DBSCAN

dbscan = DBSCAN(eps=0.5, min_samples=5)
cluster_labels = dbscan.fit_predict(X)
```

## Dimensionality Reduction

### Principal Component Analysis (PCA)
```python
from sklearn.decomposition import PCA

pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)
explained_variance = pca.explained_variance_ratio_
```

### t-SNE
```python
from sklearn.manifold import TSNE

tsne = TSNE(n_components=2, random_state=42)
X_tsne = tsne.fit_transform(X)
```

### Linear Discriminant Analysis
```python
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

lda = LinearDiscriminantAnalysis(n_components=2)
X_lda = lda.fit_transform(X, y)
```

## Model Evaluation

### Classification Metrics
```python
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix, roc_auc_score
)

# Basic metrics
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='weighted')
recall = recall_score(y_test, y_pred, average='weighted')
f1 = f1_score(y_test, y_pred, average='weighted')

# Detailed report
report = classification_report(y_test, y_pred)
cm = confusion_matrix(y_test, y_pred)

# ROC AUC (for binary/multiclass)
auc = roc_auc_score(y_test, y_proba, multi_class='ovr')
```

### Regression Metrics
```python
from sklearn.metrics import (
    mean_squared_error, mean_absolute_error, r2_score
)

mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
rmse = np.sqrt(mse)
```

## Cross-Validation
```python
from sklearn.model_selection import cross_val_score, cross_validate

# Simple cross-validation
scores = cross_val_score(clf, X, y, cv=5)
print(f"CV scores: {scores.mean():.3f} (+/- {scores.std() * 2:.3f})")

# Cross-validation with multiple metrics
scoring = ['accuracy', 'precision_macro', 'recall_macro']
cv_results = cross_validate(clf, X, y, cv=5, scoring=scoring)
```

## Hyperparameter Tuning

### Grid Search
```python
from sklearn.model_selection import GridSearchCV

param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [3, 5, 10],
    'min_samples_split': [2, 5, 10]
}

grid_search = GridSearchCV(
    RandomForestClassifier(random_state=42),
    param_grid,
    cv=5,
    scoring='accuracy',
    n_jobs=-1
)
grid_search.fit(X_train, y_train)

best_params = grid_search.best_params_
best_score = grid_search.best_score_
```

### Randomized Search
```python
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import randint

param_dist = {
    'n_estimators': randint(10, 100),
    'max_depth': randint(1, 10)
}

random_search = RandomizedSearchCV(
    RandomForestClassifier(random_state=42),
    param_distributions=param_dist,
    n_iter=20,
    cv=5,
    random_state=42
)
random_search.fit(X_train, y_train)
```

## Feature Engineering

### Feature Selection
```python
from sklearn.feature_selection import (
    SelectKBest, f_classif, RFE, SelectFromModel
)

# Univariate selection
selector = SelectKBest(score_func=f_classif, k=10)
X_selected = selector.fit_transform(X, y)

# Recursive Feature Elimination
rfe = RFE(estimator=LogisticRegression(), n_features_to_select=10)
X_rfe = rfe.fit_transform(X, y)

# Model-based selection
sfm = SelectFromModel(RandomForestClassifier())
X_sfm = sfm.fit_transform(X, y)
```

### Polynomial Features
```python
from sklearn.preprocessing import PolynomialFeatures

poly = PolynomialFeatures(degree=2, include_bias=False)
X_poly = poly.fit_transform(X)
```

## Pipelines
```python
from sklearn.pipeline import Pipeline, make_pipeline

# Create pipeline
pipe = Pipeline([
    ('scaler', StandardScaler()),
    ('selector', SelectKBest(k=10)),
    ('classifier', RandomForestClassifier())
])

# Alternative syntax
pipe = make_pipeline(
    StandardScaler(),
    SelectKBest(k=10),
    RandomForestClassifier()
)

# Fit and predict
pipe.fit(X_train, y_train)
y_pred = pipe.predict(X_test)

# Grid search with pipeline
param_grid = {
    'selector__k': [5, 10, 15],
    'classifier__n_estimators': [50, 100]
}
grid_search = GridSearchCV(pipe, param_grid, cv=5)
```

## Advanced Features

### Column Transformer
```python
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder

# For mixed data types
numeric_features = ['age', 'fare']
categorical_features = ['sex', 'class']

preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numeric_features),
        ('cat', OneHotEncoder(), categorical_features)
    ]
)

# Complete pipeline
clf = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', LogisticRegression())
])
```

### Ensemble Methods
```python
from sklearn.ensemble import VotingClassifier, BaggingClassifier

# Voting classifier
estimators = [
    ('lr', LogisticRegression()),
    ('rf', RandomForestClassifier()),
    ('svc', SVC(probability=True))
]
voting_clf = VotingClassifier(estimators, voting='soft')

# Bagging
bagging_clf = BaggingClassifier(
    base_estimator=DecisionTreeClassifier(),
    n_estimators=10,
    random_state=42
)
```

### Model Persistence
```python
import joblib
import pickle

# Save model
joblib.dump(clf, 'model.pkl')

# Load model
loaded_model = joblib.load('model.pkl')

# Alternative with pickle
with open('model.pkl', 'wb') as f:
    pickle.dump(clf, f)

with open('model.pkl', 'rb') as f:
    loaded_model = pickle.load(f)
```

## Working with Text Data

### Text Preprocessing
```python
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer

# Bag of Words
count_vec = CountVectorizer(max_features=1000, stop_words='english')
X_counts = count_vec.fit_transform(text_data)

# TF-IDF
tfidf_vec = TfidfVectorizer(max_features=1000, stop_words='english')
X_tfidf = tfidf_vec.fit_transform(text_data)
```

## Model Interpretation

### Feature Importance
```python
# For tree-based models
importance = clf.feature_importances_
feature_names = iris.feature_names
feature_importance = pd.DataFrame({
    'feature': feature_names,
    'importance': importance
}).sort_values('importance', ascending=False)

# For linear models
coefficients = lr.coef_[0]
```

### Permutation Importance
```python
from sklearn.inspection import permutation_importance

perm_importance = permutation_importance(clf, X_test, y_test, n_repeats=10, random_state=42)
```

## Common Pitfalls and Best Practices

### Data Leakage Prevention
```python
# Correct: fit on training, transform on both
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)  # Don't fit on test!

# Use pipelines to prevent leakage
pipe = Pipeline([
    ('scaler', StandardScaler()),
    ('model', LogisticRegression())
])
```

### Handling Imbalanced Data
```python
from sklearn.utils.class_weight import compute_class_weight

# Class weights
class_weights = compute_class_weight('balanced', classes=np.unique(y), y=y)
clf = RandomForestClassifier(class_weight='balanced')

# SMOTE (requires imbalanced-learn)
# from imblearn.over_sampling import SMOTE
# smote = SMOTE(random_state=42)
# X_resampled, y_resampled = smote.fit_resample(X, y)
```

### Version Information
```python
import sklearn
sklearn.show_versions()  # Display versions of sklearn and dependencies
```

## Performance Tips

- Use `n_jobs=-1` for parallel processing where available
- Use `sparse` matrices for high-dimensional sparse data
- Consider feature scaling for distance-based algorithms
- Use `random_state` parameters for reproducibility
- Profile your code and use appropriate algorithms for data size
- Consider using `partial_fit` for online learning with large datasets