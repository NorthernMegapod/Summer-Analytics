# --- Enhanced Preprocessing & Feature Engineering ---
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from scipy.ndimage import gaussian_filter1d

# Load train and test data
train = pd.read_csv('hacktrain.csv')
test = pd.read_csv('.csv')

# Prepare features and target
X_train = train.drop(['Unnamed: 0', 'ID', 'class'], axis=1)
y_train = train['class']
X_test = test.drop(['Unnamed: 0', 'ID'], axis=1)

# Impute missing values
imputer = SimpleImputer(strategy='median')
X_train_imp = imputer.fit_transform(X_train)
X_test_imp = imputer.transform(X_test)

# Denoising with Gaussian smoothing
X_train_smooth = gaussian_filter1d(X_train_imp, sigma=1.5, axis=1)
X_test_smooth = gaussian_filter1d(X_test_imp, sigma=1.5, axis=1)

# Feature engineering: statistics and slope
def add_features(X):
    mean = X.mean(axis=1).reshape(-1,1)
    std = X.std(axis=1).reshape(-1,1)
    slope = np.array([np.polyfit(np.arange(X.shape[1]), row, 1)[0] for row in X]).reshape(-1,1)
    return np.hstack([X, mean, std, slope])

X_train_eng = add_features(X_train_smooth)
X_test_eng = add_features(X_test_smooth)

# --- Feature Selection & Model Tuning ---
from sklearn.feature_selection import SelectFromModel
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV

# L1-based feature selection
selector = SelectFromModel(
    LogisticRegression(penalty='l1', solver='liblinear', multi_class='ovr', C=0.1, max_iter=1000)
)
X_train_sel = selector.fit_transform(X_train_eng, y_train)
X_test_sel = selector.transform(X_test_eng)

# Hyperparameter grid
param_grid = {
    'C': [0.001, 0.01, 0.1, 1, 10],
    'penalty': ['l1', 'l2'],
    'class_weight': [None, 'balanced']
}

# Grid search with weighted precision
grid = GridSearchCV(
    LogisticRegression(solver='liblinear', multi_class='ovr', max_iter=1000),
    param_grid,
    cv=5,
    scoring='precision_weighted'
)
grid.fit(X_train_sel, y_train)

# --- Prediction on Test Set ---
y_test_pred = grid.best_estimator_.predict(X_test_sel)

# --- Submission File Creation ---
submission = pd.DataFrame({
    'ID': test['ID'],
    'class': y_test_pred
})

submission.to_csv('submission.csv', index=False)

