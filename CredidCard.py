import numpy as np
import pandas as pd
from collections import Counter
from imblearn.over_sampling import ADASYN
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, precision_score,
                            recall_score, f1_score,
                            confusion_matrix, classification_report)

# 1. Load data and take 10% stratified sample
data = pd.read_csv("D:\\Datasets\\CreditCard\\creditcard.csv")

# Stratified sampling - maintains original class distribution
sample_data = data.groupby('Class', group_keys=False).apply(
    lambda x: x.sample(frac=0.1, random_state=42)
)

print(f"Working with {len(sample_data):,} samples ({len(sample_data)/len(data)*100:.1f}% of original data)")
print("Sample class distribution:\n", sample_data['Class'].value_counts())

# 2. Separate features and target
X = sample_data.drop('Class', axis=1)
y = sample_data['Class']

# 3. Standardize only Amount and Time (other features are already transformed)
scaler = StandardScaler()
X[['Time', 'Amount']] = scaler.fit_transform(X[['Time', 'Amount']])

# 4. Apply ADASYN with reduced sampling strategy for efficiency
adasyn = ADASYN(sampling_strategy=0.5, random_state=42)  # Only oversample to 50% of majority class
X_resampled, y_resampled = adasyn.fit_resample(X, y)

print("\nClass distribution after ADASYN:")
print("Original:", Counter(y))
print("Resampled:", Counter(y_resampled))

# 5. Train-test split with stratification
X_train, X_test, y_train, y_test = train_test_split(
    X_resampled, y_resampled,
    test_size=0.2,
    random_state=42,
    stratify=y_resampled
)

# 6. Initialize optimized Random Forest
rfc = RandomForestClassifier(
    n_estimators=150,  # Reduced from original 500
    max_depth=15,      # Shallower trees
    n_jobs=-1,         # Parallel processing
    random_state=42,
    class_weight='balanced'  # Additional protection against imbalance
)

# 7. Simplified hyperparameter grid for faster tuning
param_dist = {
    'n_estimators': [100, 150, 200],
    'max_depth': [10, 15, 20, None],
    'min_samples_split': [2, 5],
    'min_samples_leaf': [1, 2]
}

# 8. Randomized Search with reduced iterations
rfc_random = RandomizedSearchCV(
    estimator=rfc,
    param_distributions=param_dist,
    n_iter=10,          # Reduced from 20
    cv=3,               # Fewer folds
    verbose=1,
    random_state=42,
    n_jobs=-1,          # Use all cores
    scoring='f1'
)

print("\nTraining model...")
rfc_random.fit(X_train, y_train)

# 9. Evaluation
y_pred = rfc_random.predict(X_test)

print("\nBest Hyperparameters:")
print(rfc_random.best_params_)

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

print("\nClassification Report:")
print(classification_report(y_test, y_pred, digits=4))

print("\nKey Metrics:")
print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
print(f"Precision: {precision_score(y_test, y_pred):.4f}")
print(f"Recall: {recall_score(y_test, y_pred):.4f}")
print(f"F1-Score: {f1_score(y_test, y_pred):.4f}")