# CreditCardFFraud
An optimized machine learning pipeline for detecting credit card fraud transactions using Random Forest with ADASYN oversampling. Designed for efficiency on large datasets while maintaining detection accuracy.
Key Features
Stratified 10% sampling for faster prototyping without losing data representativeness

ADASYN oversampling to handle extreme class imbalance (fraud vs. non-fraud)

Optimized Random Forest with hyperparameter tuning via RandomizedSearchCV

Efficient memory usage through feature standardization and downsampling

Comprehensive evaluation with precision, recall, F1-score, and confusion matrix

Dataset
Uses the Kaggle Credit Card Fraud Detection Dataset:

284,807 transactions (492 frauds → 0.172% imbalance)--> USE Sampling technique for class imbalance (ADASYN)

30 features (PCA transformed + Time and Amount)

Performance Optimizations
10x faster processing via:

Stratified 10% subsampling

Reduced ADASYN sampling strategy (sampling_strategy=0.5)

Simplified Random Forest (150 trees vs. 500)

Memory efficiency through:

Selective feature standardization (only Time and Amount)

Optimized hyperparameter search (10 iterations vs. 20)
